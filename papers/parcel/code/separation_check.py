"""Verify the seed-set separation bound by brute force on small instances.

THE CLAIM (Proposition 1 in THEORY.md)
--------------------------------------
Classical influence maximization chooses ONE set and lets it serve the
whole network. PARCEL's setting allocates a possibly different set to
each receiver. The claim is that this difference is not cosmetic: at an
identical budget the seed-set formulation can be worse than
per-receiver allocation by a factor Theta(n) in the number of
receivers.

Construction: n receivers, n distinct items, receiver i is satisfied
only by item i, every item costs c.

  per-receiver at budget B = n*c: send item i to receiver i.
      cost n*c, utility n.
  seed set   at budget B = n*c: one set S goes to ALL n receivers, so it
      costs n*|S|*c; the budget forces |S| <= 1, and a single item
      satisfies exactly one receiver.
      cost n*c, utility 1.

So the ratio is exactly n. This script confirms it by exhaustive search
over both formulations rather than trusting the algebra, and also
checks the variant where broadcast is FREE (billed once, not per
receiver) but attention saturates -- there the seed set is punished by
over-delivery instead of by cost, so the separation survives for a
different reason.

Run: python3 separation_check.py
"""

from itertools import combinations, product


def utility_plain(alloc, n):
    """Receiver i is satisfied iff item i reached it. No saturation."""
    return sum(1 for i in range(n) if i in alloc[i])


def utility_saturating(alloc, n, free_slots=1, penalty=1.0):
    """As above, minus a penalty for each item beyond the receiver's budget.

    Models the case where transmission is not billed but attention is:
    loading a receiver past `free_slots` items costs `penalty` each.
    """
    total = 0.0
    for i in range(n):
        got = 1.0 if i in alloc[i] else 0.0
        excess = max(0, len(alloc[i]) - free_slots)
        total += got - penalty * excess
    return total


def best_per_receiver(n, cost, budget, util):
    """Exhaustive search over independent per-receiver allocations."""
    items = range(n)
    best = float("-inf")
    # Each receiver independently gets any subset; the budget couples them,
    # so enumerate joint choices over the (small) subset lattice.
    subsets = [frozenset(s) for r in range(n + 1)
               for s in combinations(items, r)]
    for combo in product(subsets, repeat=n):
        spend = sum(len(s) * cost for s in combo)
        if spend > budget:
            continue
        best = max(best, util(list(combo), n))
    return best


def best_seed_set(n, cost, budget, util, billed_per_receiver=True):
    """Exhaustive search over ONE set broadcast to every receiver."""
    items = range(n)
    best = float("-inf")
    for r in range(n + 1):
        for s in combinations(items, r):
            s = frozenset(s)
            spend = len(s) * cost * (n if billed_per_receiver else 1)
            if spend > budget:
                continue
            best = max(best, util([s] * n, n))
    return best


def main():
    cost = 1.0
    print("Billed per receiver (the cost argument):")
    print(f"{'n':>3s} {'budget':>7s} {'per-receiver':>13s} {'seed set':>9s} "
          f"{'ratio':>6s}")
    for n in range(1, 6):
        budget = n * cost
        pr = best_per_receiver(n, cost, budget, utility_plain)
        ss = best_seed_set(n, cost, budget, utility_plain)
        ratio = pr / ss if ss > 0 else float("inf")
        flag = "" if abs(ratio - n) < 1e-9 else "   <-- MISMATCH"
        print(f"{n:3d} {budget:7.1f} {pr:13.1f} {ss:9.1f} {ratio:6.2f}{flag}")

    print("\nBroadcast free, attention saturates (the saturation argument);")
    print("budget is not binding, so any loss is over-delivery alone:")
    print(f"{'n':>3s} {'per-receiver':>13s} {'seed set':>9s}")
    for n in range(1, 6):
        big = 10.0 * n * n           # budget deliberately non-binding
        pr = best_per_receiver(n, cost, big, utility_saturating)
        ss = best_seed_set(n, cost, big, utility_saturating,
                           billed_per_receiver=False)
        print(f"{n:3d} {pr:13.1f} {ss:9.1f}")

    print("\nExpected: ratio == n in the first table; in the second the")
    print("seed set stops improving because every extra item it broadcasts")
    print("lands in n-1 receivers that do not need it.")


if __name__ == "__main__":
    main()
