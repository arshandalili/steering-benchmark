from pathlib import Path

from steering_benchmark import config as config_lib


def test_smoke_config_loads():
    repo_root = Path(__file__).resolve().parents[1]
    exp_path = repo_root / "configs" / "experiments" / "smoke_tinygpt2.yaml"
    registry_dir = repo_root / "configs" / "registry"
    resolved = config_lib.resolve_experiment(exp_path, registry_dir)
    assert resolved["model"]["hf_id"] == "sshleifer/tiny-gpt2"
