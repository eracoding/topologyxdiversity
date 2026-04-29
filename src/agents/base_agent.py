import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from vllm import SamplingParams

from src.utils.json_parser import parse_agent_output
from src.utils.cost_tracker import CostTracker

if TYPE_CHECKING:
    from vllm import LLM


@dataclass
class AgentOutput:
    emotions: dict[str, int]
    confidence: dict[str, float]
    reasoning: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_success: bool = True


class BaseAgent(ABC):
    """Abstract base class for agents"""

    def __init__(
        self,
        name: str,
        llm: "LLM",
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 42,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.name = name
        self.model = model
        self.cost_tracker = cost_tracker

        self.llm = llm
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Returning system prompt for the agent"""
        ...
    
    def predict(self, user_message: str, cot_block: str = "") -> AgentOutput:
        """Send message to LLM and parse the response"""
        system_prompt = self.get_system_prompt()
        if system_prompt and cot_block:
            system_prompt = system_prompt + '\n\n' + cot_block
        elif cot_block:
            system_prompt = cot_block
        
        start = time.time()
        outputs = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            sampling_params=self.sampling_params,
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
            # Return default all-zeros on parse failure
            from src.data.loader import EMOTIONS
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
