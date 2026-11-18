"""Padded FULL-state tables and the masked SetNet (EXPERIMENT_B.md 3.5, 10 M17; fixes
REVIEW_B 9.5 and 9.6): equivalence with A's SetNet, pad invariance, mixed-k batches,
no future leakage, and that importing expb leaves A's fast-path flag and outputs alone."""
import importlib

import numpy as np
import pytest
import torch

from transvrp.expa.features import G, F, RouteCache, RouteStats, stage_batch
from transvrp.expa.model import NetConfig, SetNet
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expb import padded
from transvrp.expb.padded import PaddedTables, SetNetMasked


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_paths(N, m, seed=1):
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))


def make_cache(N=60, m=5, seed=1):
    spec = make_spec(4.0, [0.6] * m, [1.0] * m, m)
    paths = make_paths(N, m, seed)
    return spec, paths, RouteCache(spec, paths, RouteStats.fit(paths))


@pytest.fixture
def no_fastpath():
    """Disable the MHA fast path for a comparison, restoring the previous flag after."""
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


def trained_like_net(seed=3, d=64):
    """A SetNet with its default random initialisation, seeded."""
    torch.manual_seed(seed)
    net = SetNet(NetConfig(d=d))
    net.eval()
    return net


def masked_from(net: SetNet) -> SetNetMasked:
    masked = SetNetMasked(NetConfig(d=net.cfg.d))
    masked.load_state_dict(net.state_dict(), strict=True)
    masked.eval()
    return masked


def random_inputs(n, T, seed=0):
    g = torch.from_numpy(np.random.default_rng(seed).normal(size=(n, G)).astype(np.float32))
    t = torch.from_numpy(np.random.default_rng(seed + 1).normal(size=(n, T, F)).astype(np.float32))
    return g, t


# --------------------------------------------------------------- equivalence (a)

def test_state_dict_of_setnet_loads_strictly_into_the_masked_net():
    net = trained_like_net()
    masked = SetNetMasked(NetConfig(d=64))
    masked.load_state_dict(net.state_dict(), strict=True)
    assert set(masked.state_dict()) == set(net.state_dict())


def test_masked_net_equals_setnet_on_unpadded_batches(no_fastpath):
    net = trained_like_net()
    masked = masked_from(net)
    glob, tok = random_inputs(9, 4)
    no_pad = torch.zeros(9, 4, dtype=torch.bool)
    with torch.no_grad():
        ref = net(glob, tok)
        out = masked(glob, tok, no_pad)
    assert out.shape == (9,)
    assert torch.allclose(out, ref, atol=1e-6)


def test_pooled_output_is_the_setnet_pooling_on_unpadded_batches(no_fastpath):
    net = trained_like_net()
    masked = masked_from(net)
    glob, tok = random_inputs(5, 3, seed=7)
    with torch.no_grad():
        p = masked.pooled(glob, tok, torch.zeros(5, 3, dtype=torch.bool))
    assert p.shape == (5, 2 * net.cfg.d)
    # SetNet's head applied to the pooled vector reproduces SetNet's output
    with torch.no_grad():
        z = torch.nn.functional.gelu(masked.head_hidden(p))
        out = torch.sigmoid(masked.head_out(z)).squeeze(-1)
        ref = net(glob, tok)
    assert torch.allclose(out, ref, atol=1e-6)


# --------------------------------------------------------------- pad invariance (b)

def test_pad_token_values_do_not_change_the_output(no_fastpath):
    masked = masked_from(trained_like_net())
    glob, tok = random_inputs(6, 5, seed=11)
    pad = torch.zeros(6, 5, dtype=torch.bool)
    pad[:, 3:] = True                       # last two tokens are padding
    tok_b = tok.clone()
    tok_b[:, 3:, :] = torch.randn(6, 2, F) * 50.0   # arbitrary pad values
    with torch.no_grad():
        out_a = masked(glob, tok, pad)
        out_b = masked(glob, tok_b, pad)
        pool_a = masked.pooled(glob, tok, pad)
        pool_b = masked.pooled(glob, tok_b, pad)
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert torch.allclose(pool_a, pool_b, atol=1e-6)


def test_masked_mean_is_the_mean_over_real_tokens_only(no_fastpath):
    masked = masked_from(trained_like_net())
    glob, tok = random_inputs(4, 4, seed=5)
    pad = torch.zeros(4, 4, dtype=torch.bool)
    pad[:, 2:] = True
    with torch.no_grad():
        pooled = masked.pooled(glob, tok, pad)
        pooled_small = masked.pooled(glob, tok[:, :2], torch.zeros(4, 2, dtype=torch.bool))
    assert torch.allclose(pooled, pooled_small, atol=1e-6)


def test_fast_path_agrees_with_the_reference_on_masked_batches():
    """Production keeps the fast path on; its masked output must match the reference."""
    masked = masked_from(trained_like_net())
    glob, tok = random_inputs(7, 4, seed=21)
    pad = torch.zeros(7, 4, dtype=torch.bool)
    pad[:, 1] = True
    prev = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.no_grad():
            fast = masked(glob, tok, pad)
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            slow = masked(glob, tok, pad)
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_fixture_restores_the_previous_fast_path_flag():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(not before)
    try:
        assert torch.backends.mha.get_fastpath_enabled() is (not before)
    finally:
        torch.backends.mha.set_fastpath_enabled(before)
    assert torch.backends.mha.get_fastpath_enabled() == before


# --------------------------------------------------------------- tables (c)

def test_table_shapes_and_pad_mask_by_stage():
    spec, paths, cache = make_cache(N=30, m=5)
    tab = PaddedTables(cache, np.arange(30))
    assert tab.glob.shape == (4, 30, G)
    assert tab.tok.shape == (4, 30, 4, F)
    ks = np.array([1, 2, 3, 4])
    glob, tok, pad = tab.batch(np.array([0, 1, 2, 3]), ks)
    assert glob.shape == (4, G) and tok.shape == (4, 4, F) and pad.shape == (4, 4)
    for i, k in enumerate(ks):
        n_real = spec.m - k
        assert pad[i].tolist() == [t >= n_real for t in range(4)]


def test_table_globals_and_real_tokens_equal_stage_batch_exactly():
    spec, paths, cache = make_cache(N=40, m=5)
    rows = np.arange(40)
    tab = PaddedTables(cache, rows)
    for k in range(1, 5):
        g_ref, t_ref = stage_batch(cache, k, rows, "full")
        glob, tok, _ = tab.batch(rows, np.full(40, k))
        assert np.array_equal(glob.numpy(), g_ref)
        assert np.array_equal(tok.numpy()[:, : spec.m - k], t_ref)


def test_mixed_k_batch_equals_per_k_unpadded_outputs(no_fastpath):
    spec, paths, cache = make_cache(N=50, m=6)
    masked = masked_from(trained_like_net(seed=9))
    tab = PaddedTables(cache, np.arange(50))
    rng = np.random.default_rng(4)
    pos = rng.integers(0, 50, size=24)
    ks = rng.integers(1, 6, size=24)
    glob, tok, pad = tab.batch(pos, ks)
    with torch.no_grad():
        mixed = masked(glob, tok, pad).numpy()
    for i in range(24):
        k = int(ks[i])
        g1, t1 = stage_batch(cache, k, np.array([pos[i]]), "full")
        with torch.no_grad():
            ref = masked(torch.from_numpy(g1), torch.from_numpy(t1),
                         torch.zeros(1, spec.m - k, dtype=torch.bool)).numpy()[0]
        assert abs(mixed[i] - ref) < 1e-6


def test_out_of_range_stage_is_refused():
    spec, paths, cache = make_cache(N=20, m=4)
    tab = PaddedTables(cache, np.arange(20))
    with pytest.raises(ValueError):
        tab.batch(np.array([0]), np.array([0]))
    with pytest.raises(ValueError):
        tab.batch(np.array([0]), np.array([4]))


# --------------------------------------------------------------- leakage

def test_stage_k_features_do_not_use_outcomes_after_stop_k():
    m, N, k = 5, 40, 2
    spec = make_spec(4.0, [0.6] * m, [1.0] * m, m)
    paths = make_paths(N, m, seed=2)
    stats = RouteStats.fit(paths)
    p2 = Paths(p=paths.p.copy(), d=paths.d.copy())
    p2.p[:, k:] = np.random.default_rng(99).uniform(0.5, 2.0, (N, m - k))
    p2.d[:, k:] = np.random.default_rng(98).uniform(0.5, 2.0, (N, m - k))
    rows = np.arange(N)
    t1 = PaddedTables(RouteCache(spec, paths, stats), rows)
    t2 = PaddedTables(RouteCache(spec, p2, stats), rows)
    ks = np.full(N, k)
    g1, tk1, _ = t1.batch(rows, ks)
    g2, tk2, _ = t2.batch(rows, ks)
    assert torch.equal(g1, g2)
    assert torch.equal(tk1, tk2)


# --------------------------------------------------------------- import (d)

def test_importing_expb_leaves_setnet_outputs_and_fast_path_flag_unchanged():
    net = trained_like_net(seed=5)
    glob, tok = random_inputs(8, 3, seed=6)
    flag = torch.backends.mha.get_fastpath_enabled()
    with torch.no_grad():
        before = net(glob, tok).numpy().copy()
    saved = dict(vars(padded))
    try:
        importlib.reload(padded)
        with torch.no_grad():
            after = net(glob, tok).numpy()
    finally:
        # reload rebinds the module's classes; put the originals back so that
        # isinstance checks in other test modules keep seeing one class
        vars(padded).update(saved)
    assert torch.backends.mha.get_fastpath_enabled() == flag
    assert np.array_equal(before, after)
