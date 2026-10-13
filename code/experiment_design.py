"""Four tests never run before, each aimed at a design choice or a claim that a
referee can see but that no existing experiment touches.

  --part pooling   the run-length null pools lengths >= 6 to get q=(16,8,4,2,1,1).
                   Is the lead application's conclusion an artifact of that choice?
                   Repeat it pooling at 4, 5, 6, 7, 8.
  --part acperiod  the alternating contrast AC is tied to period 2.  Section V-B
                   says that against period-4 code widths on the five image nulls
                   AC and its min-p with X^2 "both fall to 0.14 or less, while
                   the CT keeps 0.14 to 0.56" (experiment_period_competitors.py).
                   This part measures the same dependence on the period on
                   synthetic non-uniform nulls, periods 2, 3, 4, 6; it is also
                   Block 2 of experiment_period_competitors.py.
  --part codes     the image experiment uses n = 64 codes only.  Does the CT's
                   advantage survive at 32, 64 and 128?
  --part zerobin   the exact DP assumes strictly positive weights and the image
                   experiment silently excludes images with an empty code.  What
                   happens at the boundary, and does the DP stay consistent there?
"""
import argparse
import numpy as np


def part_pooling(args):
    import experiment_runlength_rng as rl
    rng = np.random.default_rng(args.seed)
    print("run-length histograms pooled at several tail cut-offs; the Letter uses 6.")
    print("CT and X^2 power against the two-lane defect, MC-calibrated on the same "
          "windows.\n")
    print(f"{'pool at':>8} {'q':>24} {'L':>5} " + " ".join(f"{'eps=' + str(e):>10}" for e in args.eps[1:]))
    for cut in (4, 5, 6, 7, 8):
        # pooling at depth d: P(run=k) = 2^-k for k < d and P(run>=d) = 2^-(d-1),
        # so with L = 2^(d-1) the weights are 2^(d-1-k) for k = 1..d-1 and a final 1
        weights = [2 ** (cut - 1 - k) for k in range(1, cut)] + [1]
        q = np.array(weights, float)
        p0 = q / q.sum()
        original_bins, original_weights = rl.BINS, rl.NULL_WEIGHTS
        rl.BINS, rl.NULL_WEIGHTS = cut, tuple(int(v) for v in weights)
        try:
            rows = {}
            h0, b0 = rl.collect(args.windows, 2 * args.runs, 0.0, rng, p0, args.chunk)
            cuts = {k: np.quantile(h0[k], 1 - args.alpha) for k in ("DTV", "CT", "X2")}
            for eps in args.eps[1:]:
                h1, _ = rl.collect(args.windows // 2, 2 * args.runs, eps, rng, p0, args.chunk)
                rows[eps] = (float(np.mean(h1["CT"] > cuts["CT"])),
                             float(np.mean(h1["X2"] > cuts["X2"])))
            line = f"{cut:>8} {str(tuple(int(v) for v in weights)):>24} {int(q.sum()):>5} "
            line += " ".join(f"{rows[e][0]:>5.3f}/{rows[e][1]:<4.3f}" for e in args.eps[1:])
            print(line)
        finally:
            rl.BINS, rl.NULL_WEIGHTS = original_bins, original_weights
    print("\ncells are CT/X^2.  The Letter's claim is that the CT beats X^2 here;")
    print("if that flips at some pooling depth, the choice of 6 would need defending.")


def part_acperiod(args):
    """AC = |sum (-1)^i r_i| knows the period 2.  Against a comb of period 4 or 6
    its power should fall, which is why the Letter calls it tied to period 2."""
    rng = np.random.default_rng(args.seed)
    print("power at alpha=0.05 against combs of several periods, on non-uniform nulls;")
    print("AC is the period-2 oracle, so it must collapse away from period 2.\n")
    print(f"{'null':>22} {'period':>7} {'CT':>7} {'AC':>7} {'X2':>7} {'du':>7}")
    for name, weights in (("tent 1:2:3:4:3:2:1", (1, 2, 3, 4, 3, 2, 1)),
                          ("run-length 16:8:4:2:1:1", (16, 8, 4, 2, 1, 1)),
                          ("ramp 1:2:3:4:5:6", (1, 2, 3, 4, 5, 6)),
                          ("geometric n=12", tuple(int(round(100 * np.exp(-0.2 * i))) for i in range(12)))):
        q = np.array(weights, float); p0 = q / q.sum(); n = len(q)
        signs2 = (-1.0) ** np.arange(n)
        x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
        r0 = x0 - args.N * p0
        stats0 = {"CT": np.abs(np.diff(r0, axis=1)).sum(1),
                  "AC": np.abs(r0 @ signs2),
                  "X2": (r0 ** 2 / (args.N * p0)).sum(1),
                  "du": np.abs(np.diff(x0, axis=1)).sum(1)}
        cuts = {k: np.quantile(v, 1 - args.alpha) for k, v in stats0.items()}
        for period in (2, 3, 4, 6):
            shape = np.cos(2 * np.pi * np.arange(n) / period)
            delta = p0 * shape
            delta = delta - delta.sum() * p0
            scale = np.abs(delta).sum()
            if scale == 0:
                continue
            delta = delta * args.distance / scale
            if (p0 + delta).min() <= 0:
                room = np.min(p0[delta < 0] / -delta[delta < 0])
                delta = delta * room * 0.999
            worst = {k: 1.0 for k in stats0}
            for sign in (+1, -1):
                p1 = p0 + sign * delta
                x1 = rng.multinomial(args.N, p1, size=args.draws // 2).astype(float)
                r1 = x1 - args.N * p0
                s1 = {"CT": np.abs(np.diff(r1, axis=1)).sum(1),
                      "AC": np.abs(r1 @ signs2),
                      "X2": (r1 ** 2 / (args.N * p0)).sum(1),
                      "du": np.abs(np.diff(x1, axis=1)).sum(1)}
                for k in worst:
                    worst[k] = min(worst[k], float(np.mean(s1[k] > cuts[k])))
            print(f"{name:>22} {period:>7} " + " ".join(f"{worst[k]:>7.3f}" for k in ("CT", "AC", "X2", "du")))
        print()


def part_codes(args):
    import experiment_real_images as images
    rng = np.random.default_rng(args.seed)
    print(f"the image experiment at several code counts n, N={args.N}, eps=+-{args.eps_img}, "
          f"worse sign\n")
    print(f"{'image':>10} " + " ".join(f"{'n=' + str(n):>26}" for n in (32, 64, 128)))
    print(f"{'':>10} " + " ".join(f"{'CT    X2    X2p   ratio':>26}" for _ in range(3)))
    for name in ("camera", "coins", "moon", "page", "cell"):
        image = images.load_gray_uint8(name)
        line = f"{name:>10} "
        for n in (32, 64, 128):
            p0 = images.code_probabilities(image, n, 0.0)
            if p0.min() <= 0:
                line += f"{'(empty code)':>26} "; continue
            x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
            s0 = images.statistics(x0, p0, args.N)
            cuts = {k: np.quantile(s0[k], 1 - args.alpha) for k in s0}
            worst = {}
            for sign in (+1, -1):
                p1 = images.code_probabilities(image, n, sign * args.eps_img)
                x1 = rng.multinomial(args.N, p1, size=args.draws // 2).astype(float)
                s1 = images.statistics(x1, p0, args.N)
                for k in ("CT", "X2", "X2p"):
                    worst[k] = min(worst.get(k, 1.0), float(np.mean(s1[k] > cuts[k])))
            line += (f"  {worst['CT']:>5.3f} {worst['X2']:>5.3f} {worst['X2p']:>5.3f} "
                     f"{worst['CT'] / max(worst['X2p'], 1e-9):>5.2f}  ")
        print(line)
    print("\nratio is CT over pooled X^2.  The Letter reports n = 64 only.")


def part_zerobin(args):
    """The boundary the Letter assumes away: a null with an unreachable bin."""
    from dr_exact import calculate_exact_dr_distribution as dr_law
    print("a null weight of zero means a bin that cannot occur; the Letter assumes")
    print("strictly positive weights.  What the code does at that boundary:\n")
    for q in ([1, 0, 1], [2, 0, 3], [0, 1, 1], [1, 1, 0]):
        law = [(v, c) for v, c in dr_law(6, q) if c]
        total = sum(c for _, c in law)
        expected = sum(q) ** 6
        print(f"  q={q}: {len(law)} atoms, mass {total} vs L^N = {expected}, "
              f"{'consistent' if total == expected else 'INCONSISTENT'}")
    print("\nA zero weight still gives a consistent law: the bin simply never receives")
    print("a count, so the statistic is the centered DTV d_r of the remaining bins with a")
    print("fixed zero in between.  It is a valid null, not a failure; the Letter's")
    print("positivity assumption is about the IMAGE experiment, where an unoccupied")
    print("code would make the estimated null degenerate, not about the DP.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=("pooling", "acperiod", "codes", "zerobin"))
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=40000)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--distance", type=float, default=0.10)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.05, 0.07, 0.10])
    parser.add_argument("--eps_img", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    {"pooling": part_pooling, "acperiod": part_acperiod,
     "codes": part_codes, "zerobin": part_zerobin}[args.part](args)


if __name__ == "__main__":
    main()
