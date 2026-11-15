"""M4: neural continuation networks (EXPERIMENT_A.md section 9, M4)."""
import numpy as np
import torch

from transvrp.expa.model import MLPNet, NetConfig, SetNet, n_params


def _batch(n=6, T=5, seed=0):
    rng = np.random.default_rng(seed)
    glob = torch.tensor(rng.uniform(-1, 1, (n, 12)), dtype=torch.float32)
    tok = torch.tensor(rng.uniform(-1, 1, (n, T, 11)), dtype=torch.float32)
    return glob, tok


def test_parameter_counts():
    torch.manual_seed(0)
    s = n_params(SetNet())
    assert 5 * 10**4 <= s <= 2 * 10**5
    # hand count in section 4.1 (review): 76,865
    assert s == 76865
    mlp = n_params(MLPNet())
    assert mlp < 5 * 10**4
    assert mlp == 35201


def test_output_shapes_and_range():
    torch.manual_seed(0)
    glob, tok = _batch()
    net = SetNet().eval()
    with torch.no_grad():
        out = net(glob, tok)
    assert out.shape == (6,)
    assert torch.all(out > 0) and torch.all(out < 1)
    mlp = MLPNet().eval()
    with torch.no_grad():
        out_m = mlp(torch.cat([glob, torch.zeros(6, 3)], dim=1))
    assert out_m.shape == (6,)
    assert torch.all(out_m > 0) and torch.all(out_m < 1)


def test_token_permutation_invariance():
    torch.manual_seed(1)
    glob, tok = _batch(n=8, T=7, seed=3)
    net = SetNet(NetConfig()).eval()
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        a = net(glob, tok)
        b = net(glob, tok[:, perm, :])
    assert float((a - b).abs().max()) < 1e-5


def test_single_token_works():
    torch.manual_seed(2)
    glob, tok = _batch(n=4, T=1, seed=7)
    net = SetNet().eval()
    with torch.no_grad():
        out = net(glob, tok)
    assert out.shape == (4,)
    assert torch.all(torch.isfinite(out))
