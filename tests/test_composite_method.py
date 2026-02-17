import torch

from steering_benchmark.methods.composite import CompositeSteering
from steering_benchmark.datasets.base import Example, BaseDataset


class DummyDataset(BaseDataset):
    def iter_group(self, group, limit=None, split=None):
        for i in range(4):
            yield Example(prompt=f"{group} {i}", target="ok", group=group, meta={})

    def groups(self):
        return ["context", "param"]


class DummyModel:
    def encode_hidden(self, prompts, layer, token_position):
        batch = len(prompts)
        return torch.ones(batch, 8) * (layer + 1)


def test_composite_fit():
    cfg = {
        "estimator": {"type": "diffmean"},
        "transform": {"normalize": True},
        "op": {"type": "add", "scale": 1.0},
        "schedule": {"layers": [1], "token_position": "last"},
    }
    method = CompositeSteering(cfg)
    intervention = method.fit(DummyModel(), DummyDataset(), {"train": {}})
    assert intervention.layer == 1
