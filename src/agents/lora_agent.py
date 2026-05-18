"""LoRA-adapter agent for B-L3 inference via vLLM."""

import time
from typing import TYPE_CHECKING, Optional

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from src.agents.base_agent import AgentOutput
from src.utils.json_parser import parse_agent_output
from src.utils.cost_tracker import CostTracker

if TYPE_CHECKING:
    from vllm import LLM


class LoRAAgent:
    """Inference agent that uses a specific QLoRA adapter via vLLM.

    Unlike PromptAgent, diversity comes from adapter parameters, not the system
    prompt. All agents share the same system prompt structure (CoT block only)
    but produce different predictions due to their specialised weights.
    """

    def __init__(
        self,
        name: str,
        llm: "LLM",
        lora_path: str,
        lora_name: str,
        lora_id: int,
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        seed: int = 42,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.name = name
        self.model = model
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.lora_request = LoRARequest(lora_name, lora_id, lora_path)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

    def get_system_prompt(self) -> str:
        """B-L3 agents have no analytical bias — CoT block is the full system prompt."""
        return ""

    def predict(self, user_message: str, cot_block: str = "") -> AgentOutput:
        """Run inference with this agent's LoRA adapter.

        Args:
            user_message: task instruction + input text
            cot_block: language-specific CoT enhancement block (used as system prompt)
        """
        from src.data.loader import EMOTIONS

        system_prompt = cot_block if cot_block else ""

        start = time.time()
        outputs = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
        )
        latency = time.time() - start

        output = outputs[0]
        raw = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens

        if self.cost_tracker:
            self.cost_tracker.record_request(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
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
