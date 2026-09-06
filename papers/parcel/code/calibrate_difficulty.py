"""Find the reasoning depth where degradation is actually measurable.

The first sweep sat at 1.00 accuracy through ~1900 tokens of confusable
distractors: the model was simply strong enough for a 3-hop chain, so
the curve was invisible. A ceiling measures nothing.

The published evidence says sensitivity to irrelevant context rises with
reasoning depth (GSM-DC fits E(m) ~ m^delta with delta increasing in
depth). So depth is the knob: pick the smallest k whose zero-distractor
accuracy is high but whose loaded accuracy has room to fall.

Target band: near-ceiling with no distractors, and clearly below ceiling
under load, so both ends of the curve are informative.

Run: python3 calibrate_difficulty.py
"""

import concurrent.futures as cf
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degradation_sweep as D  # noqa: E402

MODEL = "gemini-3.5-flash-lite"
LEVELS = [0, 1000, 4000]
DEPTHS = [4, 6, 8, 10]
N = 8


def probe(key, k, words, n=N):
    def one(i):
        rng = random.Random(f"cal|{k}|{words}|{i}")
        prompt, answer, _ = D.build_prompt(k, "confusable", words, rng)
        return D.graded(D.call_gemini(MODEL, prompt, key), answer)

    with cf.ThreadPoolExecutor(3) as ex:
        got = [x for x in ex.map(one, range(n)) if x is not None]
    return sum(got) / len(got) if got else float("nan")


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    print(f"model={MODEL}  n={N} per cell")
    print("k_hops " + "".join(f"{'w=' + str(w):>10s}" for w in LEVELS))
    for k in DEPTHS:
        row = [probe(key, k, w) for w in LEVELS]
        print(f"{k:6d} " + "".join(f"{v:10.2f}" for v in row))
    print("\nPick the smallest depth that is near-ceiling at w=0 and "
          "clearly below ceiling at w=4000.")


if __name__ == "__main__":
    main()
