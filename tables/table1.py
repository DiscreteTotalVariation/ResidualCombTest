"""Table I body: exact size and power at N = 40, both comb signs at identical l1,
all five nulls of experiment_exact_power.py, with the exact collapse index of
each null (results/experiment_exact_index.txt) so the table shows the index
ordering the uncentered d_u's loss.  Emits LaTeX rows to stdout and to paper/table1.tex.
Higher power is better; size is the attained level, at most alpha = 0.05 by
construction."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "experiment_exact_power.txt"
IDX = ROOT / "results" / "experiment_exact_index.txt"
OUT = ROOT / "paper" / "table1.tex"

# in order of the collapse index, so the uncentered d_u's loss reads down the table
ROWS = [("uniform control", "flat"),
        ("ramp", "ramp"),
        ("tent", "tent"),
        ("run-length", "run-length"),
        ("comb null", "comb")]
# one row per null; the two comb signs side by side as "+ / -" so the sign
# asymmetry of the uncentered d_u reads across a row and the index ordering down a column
HEAD = r"""\begin{tabular}{@{}lc cc cc@{}}
\toprule
 & & \multicolumn{2}{c}{CT} & \multicolumn{2}{c}{$d_{\mathbf u}$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
null & index & size & power $+$\,/\,$-$ & size & power $+$\,/\,$-$ \\
\midrule
"""
TAIL = r"""\bottomrule
\end{tabular}"""
N = 40


def parse():
    text = SRC.read_text().splitlines()
    table = {}
    current = None
    for line in text:
        m = re.match(r"^(\S.*?)\s+q0=.*phase (\+|-)", line)
        if m:
            current = (m.group(1).strip(), m.group(2))
            continue
        parts = line.replace("|", " ").split()
        if current and parts and parts[0] == str(N):
            nums = parts[1:]
            # CT crit size power | DTV crit size power | DTV2 region size power | DTV2L size power
            rc, rs, rp = nums[0], nums[1], nums[2]
            dc, ds, dp = nums[3], nums[4], nums[5]
            region, s2, p2 = nums[6], nums[7], nums[8]
            sl, pl = nums[9], nums[10]
            table[current] = dict(rc=int(rc), rs=float(rs), rp=float(rp), dc=int(dc),
                                  ds=float(ds), dp=float(dp), s2=float(s2), p2=float(p2),
                                  sl=float(sl), pl=float(pl))
    return table


def fmt(x):
    return f"{x:.3f}"


def indices():
    """name -> exact collapse index at N from experiment_exact_index.txt."""
    out, current = {}, None
    for line in IDX.read_text().splitlines():
        m = re.match(r"^(\S.*?)\s+q=[\d:]+$", line)
        if m:
            current = m.group(1).strip()
            continue
        parts = line.replace("|", " ").split()
        if current and parts and parts[0] == str(N):
            out[current] = float(parts[1])
    return out


def main():
    t = parse()
    idx = indices()
    lines = []
    def pair(key_p, key_s, name):
        cells = []
        for phase in "+-":
            r = t[(name, phase)]
            flag = r"$^{\dagger}$" if r[key_p] < r[key_s] else ""
            cells.append(fmt(r[key_p]) + flag)
        return r"\,/\,".join(cells)

    for name, q in ROWS:
        r = t[(name, "+")]              # sizes do not depend on the sign
        lines.append(f"{q} & {idx[name]:.2f} & {fmt(r['rs'])} & {pair('rp', 'rs', name)} & "
                     f"{fmt(r['ds'])} & {pair('dp', 'ds', name)} \\\\")
    sizes2 = sorted(t[(name, "+")]["s2"] for name, _ in ROWS)
    print(f"two-sided d_u, repository only (not tabulated): sizes {fmt(sizes2[0])} to {fmt(sizes2[-1])}")
    body = HEAD + "\n".join(lines) + "\n" + TAIL
    OUT.write_text(body)
    print(body)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
