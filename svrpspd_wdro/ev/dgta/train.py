"""train.py — REINFORCE training for the DGTA-RL baseline, Algorithm 1
of Chen, Imdahl, Lai & Van Woensel (2025):

  1. sample a batch of instances
  2. roll out the CURRENT policy (sampling actions) -> tour cost L
  3. roll out the BASELINE policy (greedy/argmax, frozen weights) on
     the same instances -> tour cost L_b
  4. REINFORCE gradient: (L - L_b) * -log p_theta(actions), i.e. the
     baseline rollout is the variance-reduction control, not part of
     the graph
  5. every `baseline_every` epochs, paired one-sided t-test comparing
     current vs baseline mean tour cost on a fixed held-out eval set;
     if the current policy wins significantly (p < 0.05), copy its
     weights into the baseline network (paper's Algorithm 1 update
     rule) — otherwise keep the old baseline another round.

Run standalone (CPU or CUDA, auto-detected):
    python -m ev.dgta.train --epochs 200 --n-customers 20 --batch 256
Or from repo root after `cd svrpspd_wdro`:
    python ev/dgta/train.py --epochs 200
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy import stats

_HERE = Path(__file__).resolve().parent
_WDRO = _HERE.parent.parent
sys.path.insert(0, str(_WDRO))

from ev.dgta.model import DGTA               # noqa: E402
from ev.dgta.env import DTSPEnv, random_instance  # noqa: E402


def rollout(model, coords, n_time, greedy, device):
    """Run one full tour for a batch of instances under `model`.
    Returns (tour_hours (B,), sum_logp (B,))."""
    B, Np1, _ = coords.shape
    N = Np1 - 1
    env = DTSPEnv(coords, n_time=n_time, device=device)
    Hbar = model.encode(coords[:, 1:, :])          # customers only, (B,N,T,D)

    logp_sum = torch.zeros(B, device=device)
    total_hours = torch.zeros(B, device=device)
    visited_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    arrival_times = torch.zeros(B, N, device=device)

    for _ in range(N):
        t_idx = env._t_idx(env.clock)
        cur_node = torch.where(env.cur > 0, env.cur - 1,
                               torch.zeros_like(env.cur))
        logits = model.step_logits(Hbar, cur_node, t_idx, arrival_times,
                                   env.clock, visited_mask)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            choice = probs.argmax(dim=-1)
        else:
            choice = torch.multinomial(probs, 1).squeeze(-1)
        logp = torch.log(probs.gather(1, choice.unsqueeze(-1))
                         .squeeze(-1).clamp(min=1e-12))
        logp_sum = logp_sum + logp

        nxt = choice + 1                            # shift back to node ids
        realized, _ = env.step(nxt)
        total_hours = total_hours + realized
        visited_mask = visited_mask.scatter(1, choice.unsqueeze(-1), True)
        arrival_times = arrival_times.scatter(
            1, choice.unsqueeze(-1), env.clock.unsqueeze(-1))

    realized = env.close_tour()
    total_hours = total_hours + realized
    return total_hours, logp_sum


def evaluate(model, coords, n_time, device):
    with torch.no_grad():
        hours, _ = rollout(model, coords, n_time, greedy=True, device=device)
    return hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-customers", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--n-time", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--baseline-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="dgta_weights.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    torch.manual_seed(args.seed)

    model = DGTA(n_time=args.n_time).to(device)
    baseline = copy.deepcopy(model).to(device)
    baseline.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_coords = random_instance(args.eval_batch, args.n_customers,
                                  device=device, seed=12345)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        coords = random_instance(args.batch, args.n_customers, device=device)

        model.train()
        L, logp = rollout(model, coords, args.n_time, greedy=False,
                          device=device)
        with torch.no_grad():
            L_b, _ = rollout(baseline, coords, args.n_time, greedy=True,
                             device=device)

        advantage = (L - L_b).detach()
        loss = (advantage * logp).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % args.baseline_every == 0 or epoch == args.epochs:
            cur_hours = evaluate(model, eval_coords, args.n_time, device)
            base_hours = evaluate(baseline, eval_coords, args.n_time, device)
            cur_np = cur_hours.cpu().numpy()
            base_np = base_hours.cpu().numpy()
            tstat, pval = stats.ttest_rel(cur_np, base_np)
            improved = cur_np.mean() < base_np.mean() and pval / 2 < 0.05
            if improved:
                baseline.load_state_dict(model.state_dict())
                baseline.eval()
                tag = "BASELINE UPDATED"
            else:
                tag = "baseline kept"
            elapsed = time.time() - t_start
            print(f"epoch {epoch:4d} | train_L={L.mean().item():.3f} "
                 f"eval_cur={cur_np.mean():.3f} eval_base={base_np.mean():.3f} "
                 f"p={pval/2:.3f} | {tag} | {elapsed:.0f}s", flush=True)

    torch.save({"model": baseline.state_dict(), "n_time": args.n_time,
               "n_customers": args.n_customers}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
