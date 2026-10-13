import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
tags = sorted(d["runs"][0]["overall"])
print("tag                  dRA_mean   per-seed dRA            fingers/step(L)  (G)   root_written(L)  (G)")
for t in tags:
    dra = [r["overall"][t] - r["overall"]["off"] for r in d["runs"]]
    w = lambda s, k: np.mean([r["written"][t][s][k] for r in d["runs"]])  # noqa: E731
    print("%-20s %+8.3f   %-22s  %6.2f        %6.2f   %6.2f        %6.2f" % (
        t, np.mean(dra), " ".join("%+.3f" % x for x in dra),
        w("zgz_local", "fingers_written_per_step"), w("zgz_global", "fingers_written_per_step"),
        w("zgz_local", "root_written_frac"), w("zgz_global", "root_written_frac")))
