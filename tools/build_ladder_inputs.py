"""Collect e(T), c(T) and T_c (specific-heat peak) of trial runs into one JSON used to design ladders.
Usage: python tools/build_ladder_inputs.py out.json dir1 dir2 ...   (later directories override earlier ones
for the same point when their lattice is larger)."""
import glob
import json
import re
import sys

import numpy as np

from polyspin.analysis.summary import load_run, thermo


def key_of(a):
    cfg = a["config_yaml"]
    cut = re.search(r"cut: (\w+)", cfg).group(1)
    par = "zG" if cut in ("A", "C", "D", "E", "E2") else "zH4"
    return f"{cut}_{int(a['dim'])}d_{par}{a[par]:g}"


def main(out, dirs):
    db = {}
    for d in dirs:
        for p in sorted(glob.glob(d + "/*.h5")):
            r = load_run(p)
            a = r["attrs"]
            if "config_yaml" not in a or "cut:" not in a["config_yaml"]:
                continue
            th = thermo(r)
            k = key_of(a)
            if k in db and db[k]["size"] > int(a["size"]):
                continue
            ic = int(np.argmax(th["c"]))
            de = np.diff(th["e"])
            db[k] = dict(size=int(a["size"]), N=int(a["N"]), init=a["config_yaml"].split("init: ")[1].split("\n")[0],
                         T=r["T"].tolist(), e=th["e"].tolist(), c=th["c"].tolist(), Tc=float(r["T"][ic]),
                         cpeak=float(th["c"][ic]), max_e_step=float(de.max()),
                         swap_min=float(r["swap"].min()), source=p.split("/")[-1])
    json.dump(db, open(out, "w"))
    for k, v in sorted(db.items()):
        print(f"{k:22s} L={v['size']:2d} Tc={v['Tc']:.3f} cpeak={v['cpeak']:.1f} max e-step={v['max_e_step']:.2f} swap_min={v['swap_min']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
