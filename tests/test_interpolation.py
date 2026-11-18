import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "select_v5", Path(__file__).parents[1] / "scripts/select_v5.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_interpolation_endpoints_and_geometry_buffers():
    parent = {"weight": torch.tensor([1.0, 2.0]), "index": torch.tensor([1, 2])}
    child = {"weight": torch.tensor([3.0, 6.0]), "index": torch.tensor([1, 2])}
    assert torch.equal(module.interpolate(parent, child, 0)["weight"], parent["weight"])
    assert torch.equal(module.interpolate(parent, child, 1)["weight"], child["weight"])
    assert torch.equal(module.interpolate(parent, child, 0.5)["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(parent["weight"], torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="nonfloating"):
        module.interpolate(parent, {**child, "index": torch.tensor([2, 3])}, 0.5)
