"""Table II body: (block 1) l1-matched comb departures on two non-uniform nulls,
N = 500; (block 2) real images, per image, N = 2000, code-width mismatch
eps = +-0.06, worst of the two signs.  Sources:
results/experiment_shape_power.txt (block 1, all columns),
results/experiment_real_images_per_image.txt and
results/experiment_rho_diagnostic.txt (block 2).  Higher power is better.
Emits LaTeX rows to stdout and to paper/table2.tex."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "table2.tex"

COLS = ["CT", "T(1)", "min-p", "X2", "X2p", "CvM", "AD", "DTV", "AC", "vN"]


def block1():
    text = (RES / "experiment_shape_power.txt").read_text().splitlines()
    header, out = None, []
    null = kind = None
    for line in text:
        m = re.match(r"^(\S.*?)\s+\[(additive|multiplicative)\]", line)
        if m:
            null, kind = m.group(1), m.group(2)
            continue
        if line.strip().startswith("shape"):
            header = line.split()[1:]
            continue
        if line.strip().startswith("comb (period 2") and null in ("tent q=1:2:3:4:3:2:1", "run-length q=16:8:4:2:1:1"):
            mirrored = "mirrored" in line
            vals = [float(v.rstrip("*")) for v in line.split()[(4 if mirrored else 3):]]
            row = dict(zip(header, vals))
            out.append((null, kind, mirrored, row))
    # worse of the two signs, statistic by statistic
    merged = {}
    for null, kind, mirrored, row in out:
        key = (null, kind)
        merged.setdefault(key, []).append(row)
    result = []
    for (null, kind), rows in merged.items():
        assert len(rows) == 2, (null, kind, len(rows))
        worse = {k: min(r[k] for r in rows) for k in rows[0]}
        result.append((null, kind, worse))
    return result


def block2():
    per = (RES / "experiment_real_images_per_image.txt").read_text().splitlines()
    img, header, rows = None, None, {}
    for line in per:
        if line.startswith("### "):
            img = line[4:].strip(); rows[img] = {}
            continue
        parts = line.split()
        if parts and parts[0] == "N":
            header = parts[2:]
            continue
        if parts and parts[0] == "2000" and img:
            eps = float(parts[1])
            rows[img][eps] = dict(zip(header, map(float, parts[2:2 + len(header)])))
    lin = {}
    block = False
    for line in (RES / "experiment_rho_diagnostic.txt").read_text().splitlines():
        if line.startswith("N = 2000"):
            block = True; continue
        if line.startswith("N = ") and block:
            break
        if block and "image" in line:
            parts = line.replace("|", " ").split()
            name = parts[1]
            nums = [float(x) for x in parts if re.match(r"^-?\d+(\.\d+)?$", x)]
            lin[name] = nums[2]        # rho_med locked linearity ...
    out = []
    for img in ("coins", "cell", "moon"):
        a, b = rows[img][0.06], rows[img][-0.06]
        worst = {k: min(a[k], b[k]) for k in a}
        out.append((img, lin[img], worst))
    return out


def block3():
    """The IEEE Std 1241 sine-wave converter test: arcsine null, capacitor mismatch."""
    text = (RES / "experiment_sine_histogram_adc.txt").read_text().splitlines()
    header, rows, lin = None, {}, None
    inblock = False
    for line in text:
        if line.startswith("N = 10000"):
            inblock = True
            lin = float(line.split("linearity of the arcsine null")[1].strip(" )"))
            continue
        if line.startswith("N = ") and inblock:
            break
        parts = line.split()
        if inblock and parts and parts[0] == "kind":
            header = parts
            continue
        if inblock and header and parts and parts[0] == "comb":
            d = dict(zip(header, parts))
            rows[float(d["eps"])] = {k: float(v) for k, v in d.items() if k != "kind"}
    a, b = rows[0.05], rows[-0.05]
    worst = {k: min(a[k], b[k]) for k in a}
    return lin, worst


def fmt(x):
    return f"{x:.3f}"


HEAD = r"""\begin{tabular}{@{}llc cccccccccc@{}}
\toprule
null & kind & index & CT & $T(1)$ & min-$p$ & $X^2$ & $X^2_p$ & CvM & AD & $d_{\mathbf u}$ & AC & vN \\
\midrule
"""
TAIL = r"""\bottomrule
\end{tabular}"""


def synthetic_index(N=500):
    out, block = {}, False
    for line in (RES / "experiment_rho_diagnostic.txt").read_text().splitlines():
        if line.startswith(f"N = {N}"):
            block = True; continue
        if line.startswith("N = ") and block:
            break
        if block and "|" in line and "null" not in line and "linearity <" not in line:
            parts = line.replace("|", " ").split()
            nums = [float(x) for x in parts if re.match(r"^-?\d+(\.\d+)?$", x)]
            label = " ".join(x for x in parts if not re.match(r"^-?\d+(\.\d+)?$", x))
            out[label] = nums[2]
    return out


def main():
    lines = []
    idx = synthetic_index()
    idx_for = {"tent q=1:2:3:4:3:2:1": idx.get("tent 1:2:3:4:3:2:1"), "run-length q=16:8:4:2:1:1": idx.get("run-length 16:8:4:2:1:1")}
    names = {"tent q=1:2:3:4:3:2:1": r"tent $1{:}2{:}3{:}4{:}3{:}2{:}1$",
             "run-length q=16:8:4:2:1:1": r"run-length $16{:}8{:}4{:}2{:}1{:}1$"}
    for null, kind, r in block1():
        cells = [r.get(c) for c in COLS]           # pooled X2 is computed only for images
        best = max(v for c, v in zip(COLS, cells) if c != "AC" and v is not None)
        row = " & ".join("--" if v is None else
                         (f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v)) for v in cells)
        ix = idx_for[null]
        lines.append(f"{names[null]} & {'add.' if kind == 'additive' else 'mult.'} & {ix:.2f} & {row} \\\\")
    lines.append(r"\midrule")
    for img, lin, w in block2():
        vals = {"CT": w["CT"], "T(1)": w["T(1)"], "min-p": w.get("minp"), "X2": w["X2"],
                "X2p": w.get("X2p"), "CvM": None, "AD": None, "DTV": w["DTV"],
                "DTV2": w["DTV2"], "AC": w["AC"], "vN": w["vN"]}
        present = [v for c, v in vals.items() if v is not None and c != "AC"]
        best = max(present)
        cells = []
        for c in COLS:
            v = vals.get(c)
            cells.append("--" if v is None else (f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v)))
        lines.append(f"{img} & width & {lin:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lin3, w3 = block3()
    vals3 = {"CT": w3["CT"], "T(1)": w3["T(1)"], "min-p": w3["min-p"],
             "X2": w3["X2"], "X2p": None, "CvM": None, "AD": None,
             "DTV": w3["DTV"], "DTV2": w3["DTV2"], "AC": w3["AC"], "vN": w3["vN"]}
    best3 = max(v for c, v in vals3.items() if v is not None and c != "AC")
    cells3 = ["--" if vals3[c] is None else
              (f"\\textbf{{{fmt(vals3[c])}}}" if vals3[c] == best3 else fmt(vals3[c]))
              for c in COLS]
    lines.append(f"converter, sine & width & {lin3:.2f} & " + " & ".join(cells3) + r" \\")
    body = HEAD + "\n".join(lines) + "\n" + TAIL + "\n"
    OUT.write_text(body)
    print(body)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
