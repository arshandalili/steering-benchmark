from steering_benchmark.core.metrics import build_metrics
from steering_benchmark.datasets.base import Example


def test_basic_metrics():
    example = Example(prompt="q", target="Paris", group="context", meta={"context": "Paris is in France"})
    metrics = build_metrics(["exact_match", "contains", "answer_in_context", "length_ratio", "refusal_match"])
    assert metrics["exact_match"].score("Paris", "Paris", example) == 1.0
    assert metrics["contains"].score("The answer is Paris", "Paris", example) == 1.0
    assert metrics["answer_in_context"].score("Paris", "Paris", example) == 1.0
    assert metrics["length_ratio"].score("Paris", "Paris", example) >= 1.0
    assert metrics["refusal_match"].score("I'm sorry, I can't help.", "Paris", example) == 1.0
