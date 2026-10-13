"""Seventh experiment: the competitor a referee will actually propose.

The Letter's order-aware competitors are the alternating contrast AC, which is
Neyman's smooth test at the TOP frequency alone and therefore an oracle for
period 2, and the von Neumann ratio.  The obvious non-oracle alternative is
missing: a smooth test with the top k frequencies, which needs no knowledge of
the period and is the standard remedy when the direction is only roughly known.
If that test matches the CT, the Letter's positioning needs care; if it pays a
real price for the extra components, the CT's case is stronger than stated.

Construction.  Under the null the standardized residual z_i = r_i / sqrt(N p_i)
has covariance I - sqrt(p) sqrt(p)^T, so the basis is built from cosine vectors,
projected off sqrt(p) and orthonormalized (Gram-Schmidt).  The statistic with k
components is the sum of the squared projections on the k HIGHEST frequencies;
k = 1 recovers AC up to the projection, and k = 2, 3, 4 are the honest versions.
Everything is Monte Carlo calibrated on the same null draws as the CT, so every
test has the same attained size.
"""
import argparse
import numpy as np


def smooth_basis(p, kmax):
    """Orthonormal vectors in the space orthogonal to sqrt(p), from the highest
    cosine frequencies downward."""
    n = len(p)
    root = np.sqrt(p)
    vectors = []
    order = list(range(n - 1, 0, -1))          # highest frequency first
    for j in order:
        v = np.cos(np.pi * j * (np.arange(n) + 0.5) / n) * root
        v = v - (v @ root) * root              # orthogonal to sqrt(p)
        for u in vectors:
            v = v - (v @ u) * u
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            continue
        vectors.append(v / norm)
        if len(vectors) == kmax:
            break
    return np.array(vectors)


def statistics(x, p0, N, basis, signs):
    r = x - N * p0
    z = r / np.sqrt(N * p0)
    projections = z @ basis.T                  # (draws, kmax)
    out = {"CT": np.abs(np.diff(r, axis=1)).sum(1),
           "AC": np.abs(r @ signs),
           "X2": (r ** 2 / (N * p0)).sum(1)}
    cumulative = np.cumsum(projections ** 2, axis=1)
    for k in range(1, basis.shape[0] + 1):
        out[f"smooth{k}"] = cumulative[:, k - 1]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=60000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--distance", type=float, default=0.04)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    nulls = [("uniform n=8", np.ones(8)),
             ("tent 1:2:3:4:3:2:1", np.array([1., 2, 3, 4, 3, 2, 1])),
             ("ramp 1:2:3:4:5:6", np.arange(1., 7)),
             ("run-length 16:8:4:2:1:1", np.array([16., 8, 4, 2, 1, 1])),
             ("geometric n=12 b=0.2", np.exp(-0.2 * np.arange(12)))]
    keys = ["CT", "AC", "X2"] + [f"smooth{k}" for k in range(1, args.kmax + 1)]
    print(f"power at alpha={args.alpha}, N={args.N}, l1={args.distance}, worse of the two")
    print(f"signs, all MC-calibrated on the same {args.draws} null draws.")
    print("smooth{k} = Neyman smooth test on the k highest frequencies; smooth1 is the top frequency only,")
    print("i.e. the period-2 oracle (AC up to the projection); smooth2..4 are the period-unknown versions;")
    print("AC = the period-2 oracle.  Comb rows are the Letter's own alternative class.\n")
    print(f"{'null':>24} {'departure':>16} " + " ".join(f"{k:>9}" for k in keys))
    for name, weights in nulls:
        p0 = weights / weights.sum()
        n = len(p0)
        basis = smooth_basis(p0, args.kmax)
        signs = (-1.0) ** np.arange(n)
        x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
        s0 = statistics(x0, p0, args.N, basis, signs)
        cuts = {k: np.quantile(s0[k], 1 - args.alpha) for k in keys}
        shapes = {"comb (period 2)": (-1.0) ** np.arange(n),
                  "comb (period 4)": np.cos(2 * np.pi * np.arange(n) / 4),
                  "tilt": np.arange(n) - (n - 1) / 2,
                  "smooth bump": np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / max(n / 6, 1)) ** 2)}
        for shape_name, shape in shapes.items():
            delta = p0 * shape
            delta = delta - delta.sum() * p0
            scale = np.abs(delta).sum()
            if scale < 1e-12:
                continue
            delta = delta * args.distance / scale
            if (p0 + delta).min() <= 0 or (p0 - delta).min() <= 0:
                room = np.min(p0[np.abs(delta) > 0] / np.abs(delta)[np.abs(delta) > 0])
                delta = delta * room * 0.999
            worst = {k: 1.0 for k in keys}
            for sign in (+1, -1):
                x1 = rng.multinomial(args.N, p0 + sign * delta,
                                     size=args.draws // 2).astype(float)
                s1 = statistics(x1, p0, args.N, basis, signs)
                for k in keys:
                    worst[k] = min(worst[k], float(np.mean(s1[k] > cuts[k])))
            best_non_oracle = max(worst[k] for k in keys if k != "AC")
            flag = "  <- CT best" if worst["CT"] >= best_non_oracle - 1e-9 else ""
            print(f"{name:>24} {shape_name:>16} "
                  + " ".join(f"{worst[k]:>9.3f}" for k in keys) + flag)
        print()
    print("If smooth2..4 matched the CT on the comb rows, the Letter would owe the")
    print("reader that comparison; if they lose power for the extra components, the")
    print("CT's advantage over non-oracle order-aware tests is real.")


if __name__ == "__main__":
    main()
