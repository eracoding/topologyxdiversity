"""Depth topology orchestrator: sequential 3-layer pipeline.

Config C: layers process the same input in sequence. Layer 1 sees only the
input text. Layer N (N > 1) sees the input + Layer N-1's full JSON output.
Final output is Layer 3's prediction — no separate aggregation step.

Supports C-L1 (stochastic: all layers use the same generic role, diversity
comes from the temperature schedule and sequential conditioning) and C-L2
(designed: Analyst → Critical Reviewer → Calibrator).

See EXPERIMENT_SPEC.md Section 4.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.agents.base_agent import AgentOutput
from src.prompts.cot_enhancement import build_cot_block
from src.prompts.inter_layer import (
    DEPTH_PROMPT_MAP,
    format_layer1_user_prompt,
    format_subsequent_layer_user_prompt,
    format_critic_user_prompt,
    format_calibrator_user_prompt,
)
from src.prompts.task_template import TASK_INSTRUCTION
from src.utils.cost_tracker import CostTracker
from src.utils.json_parser import parse_agent_output
from src.data.loader import EMOTIONS

import time


@dataclass
class LayerConfig:
    name: str
    role: str               # "stochastic" | "analyst" | "critic" | "calibrator" | "initial_classifier" | "error_corrector"
    system_prompt: str      # raw string (empty = no designed role) or key into DEPTH_PROMPT_MAP
    sampling_params: SamplingParams
    lora_request: Optional[LoRARequest] = None   # set for C-L3; None for C-L1/L2


class DepthTopology:
    """Config C: sequential 3-layer pipeline with inter-layer conditioning."""

    def __init__(
        self,
        llm: LLM,
        layers: list[LayerConfig],
        reasoning_banks_path: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.llm = llm
        self.layers = layers
        self.reasoning_banks_path = reasoning_banks_path
        self.cost_tracker = cost_tracker

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_many(self, texts: list[str], language: str = "") -> list[dict]:
        """Run the 3-layer pipeline on a batch of texts"""
        cot_block = build_cot_block(language, self.reasoning_banks_path) if language else ""
        n = len(texts)

        layer_outputs_per_sample: list[list[AgentOutput]] = [[] for _ in range(n)]
        prior_outputs: list[Optional[dict]] = [None] * n

        for i, layer in enumerate(self.layers):
            designed_system = DEPTH_PROMPT_MAP.get(layer.system_prompt, layer.system_prompt)
            if designed_system and cot_block:
                system_prompt = designed_system + "\n\n" + cot_block
            elif cot_block:
                system_prompt = cot_block
            else:
                system_prompt = designed_system

            conversations = []
            for idx, text in enumerate(texts):
                prior = prior_outputs[idx]
                if i == 0 or prior is None:
                    user_message = format_layer1_user_prompt(text, TASK_INSTRUCTION)
                elif layer.role == "critic":
                    user_message = format_critic_user_prompt(text, prior, TASK_INSTRUCTION)
                elif layer.role == "calibrator":
                    user_message = format_calibrator_user_prompt(text, prior, TASK_INSTRUCTION)
                else:
                    user_message = format_subsequent_layer_user_prompt(text, prior, TASK_INSTRUCTION, layer.name)
                conversations.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ])

            chat_kwargs = dict(messages=conversations, sampling_params=layer.sampling_params)
            if layer.lora_request is not None:
                chat_kwargs["lora_request"] = layer.lora_request

            start = time.time()
            outputs = self.llm.chat(**chat_kwargs)
            latency_per = (time.time() - start) / n

            for idx, output in enumerate(outputs):
                raw = output.outputs[0].text
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)

                if self.cost_tracker:
                    self.cost_tracker.record_request(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency=latency_per,
                    )

                parsed = parse_agent_output(raw)
                if parsed is None:
                    ao = AgentOutput(
                        emotions={e: 0 for e in EMOTIONS},
                        confidence={e: 0.5 for e in EMOTIONS},
                        reasoning="[PARSE FAILURE]",
                        raw_response=raw,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        parse_success=False,
                    )
                else:
                    ao = AgentOutput(
                        emotions=parsed["emotions"],
                        confidence=parsed["confidence"],
                        reasoning=parsed["reasoning"],
                        raw_response=raw,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        parse_success=True,
                    )

                layer_outputs_per_sample[idx].append(ao)
                prior_outputs[idx] = {
                    "emotions": ao.emotions,
                    "confidence": ao.confidence,
                    "reasoning": ao.reasoning,
                }

        results = []
        for los in layer_outputs_per_sample:
            final = los[-1]
            results.append({
                "emotions": final.emotions,
                "confidence": final.confidence,
                "layer_outputs": los,
                "parse_failures": sum(1 for o in los if not o.parse_success),
            })
        return results

    def predict_single(self, text: str, language: str = "") -> dict:
        """Run the 3-layer pipeline on a single text.

        Returns dict with:
            - emotions: Layer 3's binary predictions (final output)
            - confidence: Layer 3's confidence scores
            - layer_outputs: list of AgentOutput for each layer
            - parse_failures: count of layers that failed to parse
        """
        cot_block = build_cot_block(language, self.reasoning_banks_path) if language else ""

        layer_outputs: list[AgentOutput] = []
        prior_output: Optional[dict] = None   # structured dict from previous layer

        for i, layer in enumerate(self.layers):
            output = self._run_layer(layer, text, cot_block, prior_output, layer_index=i)
            layer_outputs.append(output)
            # Pass the full structured output (emotions + confidence + reasoning) to next layer
            prior_output = {
                "emotions": output.emotions,
                "confidence": output.confidence,
                "reasoning": output.reasoning,
            }

        final = layer_outputs[-1]
        parse_failures = sum(1 for o in layer_outputs if not o.parse_success)

        return {
            "emotions": final.emotions,
            "confidence": final.confidence,
            "layer_outputs": layer_outputs,
            "parse_failures": parse_failures,
        }

    def _run_layer(
        self,
        layer: LayerConfig,
        text: str,
        cot_block: str,
        prior_output: Optional[dict],
        layer_index: int,
    ) -> AgentOutput:
        """Build the prompt for one layer and call the LLM."""
        # Resolve system prompt: empty = CoT block only
        designed_system = DEPTH_PROMPT_MAP.get(layer.system_prompt, layer.system_prompt)
        if designed_system and cot_block:
            system_prompt = designed_system + "\n\n" + cot_block
        elif cot_block:
            system_prompt = cot_block
        else:
            system_prompt = designed_system

        # Build user message based on layer role and position
        if layer_index == 0 or prior_output is None:
            user_message = format_layer1_user_prompt(text, TASK_INSTRUCTION)
        elif layer.role == "critic":
            user_message = format_critic_user_prompt(text, prior_output, TASK_INSTRUCTION)
        elif layer.role == "calibrator":
            user_message = format_calibrator_user_prompt(text, prior_output, TASK_INSTRUCTION)
        else:
            # Generic stochastic conditioning (C-L1 Layers 2 and 3)
            user_message = format_subsequent_layer_user_prompt(
                text, prior_output, TASK_INSTRUCTION, layer.name
            )

        start = time.time()
        chat_kwargs = dict(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            sampling_params=layer.sampling_params,
        )
        if layer.lora_request is not None:
            chat_kwargs["lora_request"] = layer.lora_request
        outputs = self.llm.chat(**chat_kwargs)
        latency = time.time() - start

        output = outputs[0]
        raw = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        if self.cost_tracker:
            self.cost_tracker.record_request(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency=latency,
            )

        parsed = parse_agent_output(raw)
        if parsed is None:
            return AgentOutput(
                emotions={e: 0 for e in EMOTIONS},
                confidence={e: 0.5 for e in EMOTIONS},
                reasoning="[PARSE FAILURE]",
                raw_response=raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                parse_success=False,
            )

        return AgentOutput(
            emotions=parsed["emotions"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            raw_response=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            parse_success=True,
        )


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def build_depth_topology(
    config: dict,
    base_config: dict,
    cost_tracker: Optional[CostTracker] = None,
) -> DepthTopology:
    """Construct a DepthTopology from YAML config dicts"""
    model_cfg = base_config["model"]
    reasoning_banks_path = base_config.get("data", {}).get("reasoning_banks_path")

    llm = LLM(
        model=model_cfg["name"],
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        dtype=model_cfg.get("dtype", "bfloat16"),
        seed=model_cfg.get("seed", 42),
        max_num_seqs=model_cfg.get("max_num_seqs", 32),
    )

    max_tokens = model_cfg.get("max_tokens", config.get("max_tokens", 512))
    layers = []
    for layer_def in config["layers"]:
        sampling = SamplingParams(
            temperature=layer_def.get("temperature", 0.7),
            top_p=layer_def.get("top_p", 0.95),
            max_tokens=max_tokens,
            seed=layer_def.get("seed", 42),
        )
        layers.append(LayerConfig(
            name=layer_def["name"],
            role=layer_def.get("role", "stochastic"),
            system_prompt=layer_def.get("system_prompt", ""),
            sampling_params=sampling,
        ))

    return DepthTopology(
        llm=llm,
        layers=layers,
        reasoning_banks_path=reasoning_banks_path,
        cost_tracker=cost_tracker,
    )


def build_lora_depth_topology(
    config: dict,
    base_config: dict,
    cost_tracker: Optional[CostTracker] = None,
) -> DepthTopology:
    """Construct a DepthTopology with per-layer LoRA adapters (C-L3).

    Each layer uses a distinct adapter trained for its sequential position.
    vLLM is launched with enable_lora=True and max_loras=num_layers.
    """
    model_cfg = base_config["model"]
    reasoning_banks_path = base_config.get("data", {}).get("reasoning_banks_path")
    n_layers = len(config["layers"])

    llm = LLM(
        model=model_cfg["name"],
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        dtype=model_cfg.get("dtype", "bfloat16"),
        seed=model_cfg.get("seed", 42),
        max_num_seqs=model_cfg.get("max_num_seqs", 32),
        enable_lora=True,
        max_loras=n_layers,
        max_lora_rank=config.get("training", {}).get("lora_rank", 8),
    )

    max_tokens = model_cfg.get("max_tokens", config.get("max_tokens", 512))
    layers = []
    for layer_def in config["layers"]:
        sampling = SamplingParams(
            temperature=layer_def.get("temperature", 0.3),
            top_p=layer_def.get("top_p", 0.95),
            max_tokens=max_tokens,
            seed=layer_def.get("seed", 42),
        )
        lora_request = LoRARequest(
            layer_def["lora_name"],
            layer_def["lora_id"],
            layer_def["lora_path"],
        )
        layers.append(LayerConfig(
            name=layer_def["name"],
            role=layer_def.get("role", "initial_classifier"),
            system_prompt=layer_def.get("system_prompt", ""),
            sampling_params=sampling,
            lora_request=lora_request,
        ))

    return DepthTopology(
        llm=llm,
        layers=layers,
        reasoning_banks_path=reasoning_banks_path,
        cost_tracker=cost_tracker,
    )
