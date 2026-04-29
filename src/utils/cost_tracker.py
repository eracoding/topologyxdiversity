import time

from dataclasses import dataclass, field

# Dataclass benefits:
# automatic method generation - __eq__, __repr__, __init__
# can make instances immutable with @dataclass(frozen=True) option
# field param allows to have default values and factory functions
# serialization with asdict method
# ordering support with order=True

@dataclass
class RequestStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0

    def __post_init__(self): # runs after init
        if self.completion_tokens < 0:
            raise ValueError("sr must be postiive")
        


@dataclass
class CostTracker:
    requests: list[RequestStats] = field(default_factory=list)
    _start_time: float = 0.0

    def start_timer(self):
        self._start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time
    
    def record_request(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_seconds: float = 0.0,
    ):
        self.requests.append(RequestStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
        ))

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)
    
    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.requests)
    
    @property
    def total_latency(self) -> float:
        return sum(r.latency_seconds for r in self.requests)

    @property
    def num_requests(self) -> int:
        return len(self.requests)
    
    def summary(self) -> dict:
        n = self.num_requests
        return {
            "num_requests": n,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_seconds": round(self.total_latency, 2),
            "wall_clock_seconds": round(self.elapsed_seconds(), 2),
            "avg_tokens_per_request": round(self.total_tokens / n, 1) if n else 0,
            "avg_latency_per_request": round(self.total_latency / n, 3) if n else 0,
        }
