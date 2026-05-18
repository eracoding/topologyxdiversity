"""Width topology orchestrator: parallel agents with aggregation.

Config B: All agents process the same input in parallel; outputs are aggregated.
Agents do NOT see each other's outputs.

Supports B-L1 (stochastic diversity, different seeds) and B-L2 (designed
diversity, different analytical bias prompts). See EXPERIMENT_SPEC.md Section 3.
"""

from typing import Optional

from src.agents.base_agent import AgentOutput
from src.agents.prompt_agent import PromptAgent
from src.aggregation.majority_vote import majority_vote, confidence_weighted_average
from src.prompts.task_template import format_user_prompt
from src.prompts.cot_enhancement import build_cot_block
from src.utils.cost_tracker import CostTracker
from src.data.loader import EMOTIONS


class WidthTopology:
    """Config B: dispatch input to N parallel agents, aggregate results."""

    def __init__(
        self,
        agents: list[PromptAgent],
        aggregation: str = "majority_vote",
        majority_threshold: int = 2,
        confidence_threshold: float = 0.5,
        reasoning_banks_path: Optional[str] = None,
    ):
        self.agents = agents
        self.aggregation = aggregation
        self.majority_threshold = majority_threshold
        self.confidence_threshold = confidence_threshold
        self.reasoning_banks_path = reasoning_banks_path

    def predict_single(self, text: str, language: str = "") -> dict:
        """Run all agents on a single text and aggregate

        Returns dict with:
            - emotions: aggregated binary predictions
            - confidence: averaged confidence scores
            - agent_outputs: list of individual AgentOutput objects
            - parse_failures: count of agents that failed to parse
            - aggregation_method: name of aggregation method used
        """
        user_message = format_user_prompt(text)
        cot_block = build_cot_block(language, self.reasoning_banks_path) if language else ""

        agent_outputs: list[AgentOutput] = []
        for agent in self.agents:
            output = agent.predict(user_message, cot_block=cot_block)
            agent_outputs.append(output)

        parse_failures = sum(1 for o in agent_outputs if not o.parse_success)

        if self.aggregation == "confidence_weighted":
            emotions, confidence = confidence_weighted_average(
                agent_outputs,
                threshold=self.confidence_threshold,
            )
            method = "confidence_weighted"
        else:
            emotions = majority_vote(
                agent_outputs,
                threshold=self.majority_threshold,
            )
            method = "majority_vote"
            n = len(agent_outputs)
            confidence = {}
            for emo in EMOTIONS:
                total = sum(o.confidence.get(emo, 0.5) for o in agent_outputs)
                confidence[emo] = round(total / n, 4) if n else 0.5

        return {
            "emotions": emotions,
            "confidence": confidence,
            "agent_outputs": agent_outputs,
            "parse_failures": parse_failures,
            "aggregation_method": method,
        }

    def predict_batch(self, texts: list[str], language: str = "") -> list[dict]:
        """Run all agents on a batch of texts using vLLM's native batched chat"""
        import time
        from src.prompts.task_template import format_user_prompt
        from src.utils.json_parser import parse_agent_output

        cot_block = build_cot_block(language, self.reasoning_banks_path) if language else ""
        n = len(texts)

        agent_outputs_all: list[list[AgentOutput]] = []

        for agent in self.agents:
            bias = agent.get_system_prompt()
            if bias and cot_block:
                system_prompt = bias + "\n\n" + cot_block
            elif cot_block:
                system_prompt = cot_block
            else:
                system_prompt = bias

            conversations = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": format_user_prompt(text)},
                ]
                for text in texts
            ]

            chat_kwargs = dict(messages=conversations, sampling_params=agent.sampling_params)
            lora_request = getattr(agent, "lora_request", None)
            if lora_request is not None:
                chat_kwargs["lora_request"] = lora_request

            start = time.time()
            outputs = agent.llm.chat(**chat_kwargs)
            latency_per = (time.time() - start) / n

            agent_outputs: list[AgentOutput] = []
            for output in outputs:
                raw = output.outputs[0].text
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)

                if agent.cost_tracker:
                    agent.cost_tracker.record_request(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency=latency_per,
                    )

                parsed = parse_agent_output(raw)
                if parsed is None:
                    agent_outputs.append(AgentOutput(
                        emotions={e: 0 for e in EMOTIONS},
                        confidence={e: 0.5 for e in EMOTIONS},
                        reasoning="[PARSE FAILURE]",
                        raw_response=raw,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        parse_success=False,
                    ))
                else:
                    agent_outputs.append(AgentOutput(
                        emotions=parsed["emotions"],
                        confidence=parsed["confidence"],
                        reasoning=parsed["reasoning"],
                        raw_response=raw,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        parse_success=True,
                    ))

            agent_outputs_all.append(agent_outputs)

        results = []
        for i in range(n):
            per_sample = [agent_outputs_all[a][i] for a in range(len(self.agents))]
            parse_failures = sum(1 for o in per_sample if not o.parse_success)

            if self.aggregation == "confidence_weighted":
                emotions, confidence = confidence_weighted_average(per_sample, threshold=self.confidence_threshold)
                method = "confidence_weighted"
            else:
                emotions = majority_vote(per_sample, threshold=self.majority_threshold)
                method = "majority_vote"
                n_agents = len(per_sample)
                confidence = {
                    emo: round(sum(o.confidence.get(emo, 0.5) for o in per_sample) / n_agents, 4)
                    for emo in EMOTIONS
                }

            results.append({
                "emotions": emotions,
                "confidence": confidence,
                "agent_outputs": per_sample,
                "parse_failures": parse_failures,
                "aggregation_method": method,
            })

        return results


def build_width_topology(
    config: dict,
    base_config: dict,
    cost_tracker: Optional[CostTracker] = None,
) -> WidthTopology:
    """Construct a WidthTopology from YAML config dicts"""
    from vllm import LLM
    from src.prompts.system_prompts import PROMPT_MAP

    model_cfg = base_config["model"]
    sampling = config.get("sampling", {})
    agg_cfg = config.get("aggregation", {})
    reasoning_banks_path = base_config.get("data", {}).get("reasoning_banks_path")

    llm = LLM(
        model=model_cfg["name"],
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        dtype=model_cfg.get("dtype", "bfloat16"),
        seed=model_cfg.get("seed", 42),
        max_num_seqs=model_cfg.get("max_num_seqs", 32),
    )

    default_seed = model_cfg.get("seed", 42)
    default_temp = sampling.get("temperature", 0.7)
    default_top_p = sampling.get("top_p", 0.95)
    max_tokens = model_cfg.get("max_tokens", config.get("max_tokens", 512))

    agents = []
    for agent_def in config["agents"]:
        name = agent_def["name"]
        # 'bias' field: empty string = no bias (B-L1), a PROMPT_MAP key, or
        # a full prompt string. If the value is a key in PROMPT_MAP, resolve it;
        # otherwise use it directly as the prompt text.
        bias_value = agent_def.get("bias", PROMPT_MAP.get(name, ""))
        bias_prompt = PROMPT_MAP.get(bias_value, bias_value) if bias_value else ""
        # Per-agent seed overrides global seed
        seed = agent_def.get("seed", default_seed)
        # Per-agent temperature override (used in depth topology; width uses global)
        temperature = agent_def.get("temperature", default_temp)
        top_p = agent_def.get("top_p", default_top_p)

        agent = PromptAgent(
            system_prompt=bias_prompt,
            name=name,
            llm=llm,
            model=model_cfg["name"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            cost_tracker=cost_tracker,
        )
        agents.append(agent)

    return WidthTopology(
        agents=agents,
        aggregation=agg_cfg.get("primary", "majority_vote"),
        majority_threshold=agg_cfg.get("majority_threshold", 2),
        confidence_threshold=agg_cfg.get("confidence_threshold", 0.5),
        reasoning_banks_path=reasoning_banks_path,
    )


def build_lora_width_topology(
    config: dict,
    base_config: dict,
    cost_tracker: Optional[CostTracker] = None,
) -> WidthTopology:
    """Construct a WidthTopology with LoRA agents for B-L3.

    Loads one vLLM instance with enable_lora=True so all adapters share the
    same GPU memory for the frozen base model.

    """
    from vllm import LLM
    from src.agents.lora_agent import LoRAAgent

    model_cfg = base_config["model"]
    sampling = config.get("sampling", {})
    agg_cfg = config.get("aggregation", {})
    reasoning_banks_path = base_config.get("data", {}).get("reasoning_banks_path")

    n_agents = len(config["agents"])

    llm = LLM(
        model=model_cfg["name"],
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        dtype=model_cfg.get("dtype", "bfloat16"),
        seed=model_cfg.get("seed", 42),
        max_num_seqs=model_cfg.get("max_num_seqs", 32),
        enable_lora=True,
        max_loras=n_agents,
        max_lora_rank=config.get("training", {}).get("lora_rank", 8),
    )

    default_seed = sampling.get("seed", model_cfg.get("seed", 42))
    default_temp = sampling.get("temperature", 0.3)
    default_top_p = sampling.get("top_p", 0.95)
    max_tokens = model_cfg.get("max_tokens", config.get("max_tokens", 512))

    agents = []
    for agent_def in config["agents"]:
        agent = LoRAAgent(
            name=agent_def["name"],
            llm=llm,
            lora_path=agent_def["lora_path"],
            lora_name=agent_def["lora_name"],
            lora_id=agent_def["lora_id"],
            model=model_cfg["name"],
            max_tokens=max_tokens,
            temperature=agent_def.get("temperature", default_temp),
            top_p=agent_def.get("top_p", default_top_p),
            seed=agent_def.get("seed", default_seed),
            cost_tracker=cost_tracker,
        )
        agents.append(agent)

    return WidthTopology(
        agents=agents,
        aggregation=agg_cfg.get("primary", "majority_vote"),
        majority_threshold=agg_cfg.get("majority_threshold", 2),
        confidence_threshold=agg_cfg.get("confidence_threshold", 0.5),
        reasoning_banks_path=reasoning_banks_path,
    )
