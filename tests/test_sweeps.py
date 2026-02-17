import torch

from steering_benchmark.core.runner import run_benchmark
from steering_benchmark.registry import register_model, register_dataset, register_method
from steering_benchmark.core.intervention import InterventionSpec, VectorAddIntervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.datasets.base import BaseDataset, Example


@register_model("dummy_model_sweep")
class DummyModel:
    def encode_hidden(self, prompts, layer, token_position):
        batch = len(prompts)
        return torch.zeros(batch, 4)

    def generate(self, prompt, intervention=None, gen_cfg=None):
        return "ok"

    @property
    def device(self):
        return torch.device("cpu")


@register_dataset("dummy_dataset_sweep")
class DummyDataset(BaseDataset):
    def iter_group(self, group, limit=None, split=None):
        for i in range(4):
            yield Example(prompt=f"{group} {i}", target="ok", group=group, meta={})

    def groups(self):
        return ["context", "param"]


@register_method("dummy_method_sweep")
class DummyMethod(SteeringMethod):
    def fit(self, model, dataset, run_cfg):
        direction = torch.zeros(4)
        intervention = VectorAddIntervention(direction=direction, scale=1.0, token_position="last")
        return InterventionSpec(layer=0, intervention=intervention)


def test_sweeps_output():
    config = {
        "name": "dummy",
        "model": {"type": "dummy_model_sweep"},
        "dataset": {"loader": "dummy_dataset_sweep"},
        "method": {"name": "dummy_method_sweep"},
        "run": {
            "eval": {"group": "context", "metrics": ["exact_match"], "max_examples": 2},
            "factors": [1.0],
            "sweeps": {"factors": [0.5, 1.0]},
        },
        "output_dir": "results",
        "registry": {"datasets": {}},
    }
    results = run_benchmark(config)
    assert "sweeps" in results
    assert "factors" in results["sweeps"]
