"""Prompt-based agent with configurable system prompt."""

from src.agents.base_agent import BaseAgent


class PromptAgent(BaseAgent):
    """An agent distinguished by its system prompt (analytical lens)."""

    def __init__(self, system_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = system_prompt

    def get_system_prompt(self) -> str:
        return self._system_prompt
