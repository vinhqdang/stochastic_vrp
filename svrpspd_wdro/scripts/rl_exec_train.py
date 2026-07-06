#!/usr/bin/env python3
"""
rl_exec_train.py — RL execution-policy baseline (Iklassov et al. 2024 style),
self-contained for Colab (torch + numpy only).

Re-implementation of the reinforcement-learning approach to stochastic VRP
of Iklassov, Sobirov, Solozabal & Takac (ACML 2024, arXiv:2311.07708),
adapted to the execution stage of the SVRPSPD: a single policy network,
trained with REINFORCE and a moving-average baseline across ALL routes,
decides continue-vs-handoff after every stop from the observable state.

State features (route-normalized, no oracle information):
    k/m, W_k/B, (B-W_k)/B, H_k/E_k, H_k/E_bar, mean & std of remaining
    increments (train-estimated) scaled by B, remaining stops fraction.

Reward: negative realized cost of the day (H_k at handoff, E_j at breach,
0 on completion) — exactly the evaluation objective.

Usage (on Colab, after uploading rl_bundle.npz):
    python rl_exec_train.py bundle=rl_bundle.npz epochs=30 out=rl_results.json
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def load_bundle(path):
    z = np.load(path, allow_pickle=True)
    n = int(z["n_routes"][0])
    routes = []
    for i in range(n):
        routes.append(dict(
            g_train=z[f"r{i}_g_train"], g_test=z[f"r{i}_g_test"],
            H=z[f"r{i}_H"], E=z[f"r{i}_E"],
            B=float(z[f"r{i}_B"][0]), inst=str(z[f"r{i}_inst"][0])))
    return routes


class Policy(nn.Module):
    def __init__(self, d_in=8, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1))

    def forward(self, x):                    # logit of P(handoff)
        return self.net(x).squeeze(-1)


def route_features(rt):
    """Static per-stop features + normalizers for one route."""
    g = rt["g_train"]
    m = g.shape[1]
    mu = g.mean(0)
    rem_mean = np.concatenate([np.cumsum(mu[::-1])[::-1][1:], [0.0]])
    rem_std = np.array([g[:, k + 1:].sum(1).std() if k + 1 < m else 0.0
                        for k in range(m)])
    return rem_mean, rem_std


def make_state(k, m, W, B, H, E, rem_mean, rem_std):
    Eb = float(E.mean())
    return np.array([
        k / m, W / B, (B - W) / B, H[k - 1] / E[k - 1], H[k - 1] / Eb,
        rem_mean[k - 1] / B, rem_std[k - 1] / max(B, 1e-9), (m - k) / m,
    ], dtype=np.float32)


def episode_batch(policy, rt, g, idx, device, greedy=False):
    """Vectorized batch of episodes over scenario rows idx. Returns realized
    costs and (logprob sums) for REINFORCE."""
    m = g.shape[1]
    B, H, E = rt["B"], rt["H"], rt["E"]
    rem_mean, rem_std = rt["_feat"]
    N = len(idx)
    W = np.zeros(N, dtype=np.float32)
    alive = np.ones(N, dtype=bool)
    costs = np.zeros(N, dtype=np.float32)
    logps = torch.zeros(N, device=device)
    for k in range(1, m + 1):
        W[alive] += g[idx[alive], k - 1]
        br = alive & (W > B)
        costs[br] = E[k - 1]
        alive &= ~br
        if k == m or not alive.any():
            break
        states = np.stack([make_state(k, m, w, B, H, E, rem_mean, rem_std)
                           for w in W[alive]])
        logits = policy(torch.as_tensor(states, device=device))
        probs = torch.sigmoid(logits)
        if greedy:
            act = (probs > 0.5).float()
        else:
            act = torch.bernoulli(probs)
        lp = torch.log(torch.where(act > 0.5, probs, 1 - probs) + 1e-9)
        idx_alive = np.where(alive)[0]
        logps[idx_alive] = logps[idx_alive] + lp
        ho = act.detach().cpu().numpy() > 0.5
        ho_rows = idx_alive[ho]
        costs[ho_rows] = H[k - 1]
        alive[ho_rows] = False
    return costs, logps


def main():
    bundle, epochs, out = "rl_bundle.npz", 30, "rl_results.json"
    batch = 256
    for a in sys.argv[1:]:
        if a.startswith("bundle="): bundle = a[7:]
        elif a.startswith("epochs="): epochs = int(a[7:])
        elif a.startswith("out="):    out = a[4:]
        elif a.startswith("batch="):  batch = int(a[6:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    routes = load_bundle(bundle)
    for rt in routes:
        rt["_feat"] = route_features(rt)
    policy = Policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    baseline = {i: 0.0 for i in range(len(routes))}

    t0 = time.time()
    for ep in range(epochs):
        ep_cost = 0.0
        order = np.random.permutation(len(routes))
        for ri in order:
            rt = routes[ri]
            g = rt["g_train"]
            idx = np.random.randint(0, g.shape[0], size=batch)
            costs, logps = episode_batch(policy, rt, g, idx, device)
            b = baseline[ri]
            adv = torch.as_tensor(costs, device=device) - b
            loss = (adv.detach() * logps).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            baseline[ri] = 0.9 * b + 0.1 * float(costs.mean())
            ep_cost += float(costs.mean())
        print(f"epoch {ep + 1}/{epochs}  mean train cost "
              f"{ep_cost / len(routes):.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    # greedy evaluation on held-out test days
    results = []
    with torch.no_grad():
        for ri, rt in enumerate(routes):
            g = rt["g_test"]
            costs, _ = episode_batch(policy, rt, g,
                                     np.arange(g.shape[0]), device,
                                     greedy=True)
            # reactive reference on the same days
            m = g.shape[1]
            W = np.cumsum(g, 1)
            br = W > rt["B"]
            ostep = np.where(br.any(1), br.argmax(1), -1)
            react = np.where(ostep >= 0, rt["E"][np.clip(ostep, 0, m - 1)], 0.0)
            results.append(dict(inst=rt["inst"], m=int(g.shape[1]),
                                rl_cost=float(costs.mean()),
                                reactive_cost=float(react.mean())))
    json.dump(results, open(out, "w"), indent=1)
    tot_rl = sum(r["rl_cost"] for r in results)
    tot_re = sum(r["reactive_cost"] for r in results)
    print(f"wrote {out}: RL total {tot_rl:.2f} vs reactive {tot_re:.2f} "
          f"(saving {100 * (tot_re - tot_rl) / max(tot_re, 1e-9):.1f}%)")


if __name__ == "__main__":
    main()
