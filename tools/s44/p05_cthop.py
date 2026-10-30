#!/usr/bin/env python3
"""S44 P0.5 (docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019): S43's asynchronous runtime at a 10 ms hop with the
event count of the gain taken since the last query instead of over the last 50 ms.

`tools/s43/async_eval.py` already converts the 50 ms gains to the hop (a_h = 1 - (1 - a_50)^(h/50)); this wrapper
only replaces `EventDrivenTracker._recent_count` (`--count since`) and otherwise runs that script unchanged, so
`--count sliding50` is the original runtime.

    python tools/s44/p05_cthop.py --count since -- --run-dir outputs/semkine/s43n_sp_s3407 --hop 10 --host-state --out X
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools" / "s43"), str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"),
                str(REPO / "tools" / "dt")]


def main():
    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--count" or argv[1] not in ("since", "sliding50"):
        raise SystemExit("usage: p05_cthop.py --count since|sliding50 -- <async_eval args>")
    mode, rest = argv[1], argv[argv.index("--") + 1:]
    hop = int(rest[rest.index("--hop") + 1]) if "--hop" in rest else 50
    import async_eval as AE
    if mode == "since":
        def since_last(self, t_ms):
            lo = t_ms - self.hop_ms + 1                      # prune exactly as the original (50 ms memory)
            for k in [k for k in self.ev_n if k < lo]:
                del self.ev_n[k]
            return int(sum(n for k, n in self.ev_n.items() if t_ms - hop < k <= t_ms))
        AE.EventDrivenTracker._recent_count = since_last
    sys.argv = ["async_eval.py"] + rest
    AE.main()


if __name__ == "__main__":
    main()
