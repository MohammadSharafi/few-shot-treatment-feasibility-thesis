"""Read results/ra3/results.json and emit the LaTeX result-macro block, then splice it
into paper_ra3/main.tex between the macro markers."""
import json, re
from pathlib import Path

RES = Path("results/ra3/results.json")
TEX = Path("paper_ra3/main.tex")


def f2(x):  # 2-dp
    return f"{x:.2f}"

def f3(x):
    return f"{x:.3f}"


def main():
    d = json.load(open(RES))
    rows = {r["mechanism"]: r for r in d["results"]}
    uq = d["qi_uniqueness_raw"]
    uqc = d["qi_uniqueness_coarse_10bin"]

    def g(name):
        return rows[name]

    macros = {
        # Use a thin space as the thousands separator to avoid literal braces in the
        # macro value (which would break the simple [^}]* splice regex).
        "NRECORDS": f"{d['n_records']:,}".replace(",", "{,}"),
        "NQI": str(len(d["qi_cols"])),
        "NCLIN": str(d["n_clinical"]),
        "FracSingleton": f"{100*uq['frac_singleton_records']:.1f}\\%",
        "FracSingletonCoarse": f"{100*uqc['frac_singleton_records']:.1f}\\%",
    }
    def put(prefix, r, sec=True):
        macros[prefix + "AUROC"] = f2(r["auroc"])
        if sec:
            macros[prefix + "AUROCsec"] = f2(r["auroc_secondary"])
        macros[prefix + "MIA"] = f2(r["mia_auc"])
        macros[prefix + "Reid"] = f3(r["reid_rate"])
        macros[prefix + "Attr"] = f2(r["attr_bacc"])

    put("Raw", g("Raw"))
    put("Kten", g("kAnon_k10"))
    put("Udp", g("UniformDP_s1.0"))
    put("Synth", g("DPSynth_s0.25"))
    put("RA", g("RA3_s1.0"))

    # Build macro block
    order = ["NRECORDS","NQI","NCLIN","FracSingleton","FracSingletonCoarse",
             "RawAUROC","RawAUROCsec","RawMIA","RawReid","RawAttr",
             "KtenAUROC","KtenAUROCsec","KtenMIA","KtenReid","KtenAttr",
             "UdpAUROC","UdpAUROCsec","UdpMIA","UdpReid","UdpAttr",
             "SynthAUROC","SynthAUROCsec","SynthMIA","SynthReid","SynthAttr",
             "RAAUROC","RAAUROCsec","RAMIA","RAReid","RAAttr"]

    tex = TEX.read_text()
    for k in order:
        if k not in macros:
            continue
        # Replace \newcommand{\Key}{...} up to the final brace on that line (greedy so
        # values that themselves contain braces, e.g. 32{,}399, splice correctly).
        pat = re.compile(r"(\\newcommand\{\\" + k + r"\}\{).*(\})\s*$", re.MULTILINE)
        tex, nsub = pat.subn(lambda m: m.group(1) + macros[k] + m.group(2), tex)
    TEX.write_text(tex)
    print("Injected macros:")
    for k in order:
        print(f"  {k:20s} {macros.get(k,'--')}")


if __name__ == "__main__":
    main()
