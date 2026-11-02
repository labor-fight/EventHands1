"""CPU tests: the single CLI of the consolidated S45-SM package (verbs, plans, the two ways to start it)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import cli  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402

CLI_FILE = P.PACKAGE / "cli.py"
PKG = f"tools.{P.PACKAGE.name}"


def test_every_verb_parses():
    examples = [["configs", "--round", "14"], ["configs", "--check"], ["freeze", "round", "--round", "15", "--tag", "T"],
                ["freeze", "files", "--name", "r15a", "--round", "15", "--tag", "T", "a", "b"],
                ["freeze", "identity", "--archive", "d", "--report", "r"], ["verify-protocol"],
                ["train", "--arm", "a", "--phase", "formal", "--config", "c"], ["train", "--config", "c"],
                ["accept", "--arm", "a", "--phase", "preflight"], ["replay", "--arm", "a"],
                ["evaluate", "--net", "n", "--mode", "e1", "--step", "5500"], ["row", "--net", "n", "--block", "b"],
                ["readouts", "--net", "n", "--task", "tl_e1"], ["bank", "run", "--arm", "a", "--shard", "2", "--shards", "6"],
                ["bank", "verify", "--arm", "a"], ["router-train", "--router", "r", "--phase", "formal"],
                ["router-decode", "--router", "r"], ["router-read", "ctx", "--router", "r"], ["select", "--id", "r6c", "--router", "r"],
                ["decide"], ["decide", "--round", "12", "--as-of-round", "11"], ["table", "--round", "12"],
                ["read", "--id", "repeat_r13"], ["jobs", "standard", "--round", "14", "--dry-run"], ["jobs", "translate", "f"],
                ["plan", "row", "--net", "n", "--block", "b"]]
    assert {cli.parse(e).verb for e in examples} == set(cli.HANDLERS)
    ns = cli.parse(["decide", "--round", "12", "--as-of-round", "11"])
    assert (ns.round, ns.as_of_round) == (12, 11) and cli.parse(["decide"]).round is None and cli.parse(["decide"]).as_of_round is None
    assert cli.parse(["evaluate", "--net", "n", "--mode", "e2"]).step == 6000


@pytest.mark.parametrize("argv", [["table", "--round", "9"], ["evaluate", "--net", "n", "--mode", "row", "--step", "4000"],
                                  ["train", "--arm", "a", "--phase", "middle", "--config", "c"], ["select", "--id", "r5"],
                                  ["read", "--id", "other"], ["accept", "--arm", "a"], ["nonexistent"]])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        cli.parse(argv)


def test_plans_resolve_from_the_registry():
    plan = cli.plan(["evaluate", "--net", "xt0_srdwa_5000", "--mode", "e1"])
    assert plan["family"] == "cell" and plan["trunk"] == "pfat0pdz" and plan["router"] == "srdwkda" and plan["register_kinds"] == ["srp", "dw", "hier"]
    plan = cli.plan(["row", "--net", "s45ref_r14", "--block", "r14"])
    assert plan["family"] == "reference" and plan["exclusive_gpu"] and plan["needs_gpu"]
    plan = cli.plan(["evaluate", "--net", "pfagtdz", "--mode", "row"])
    assert plan["family"] == "oracle" and plan["register_kinds"] == []
    plan = cli.plan(["readouts", "--net", "pfat0pdz_rep_5500", "--task", "tl_e1"])
    assert plan["family"] == "router" and plan["arm"] == "pfat0pdz_rep" and plan["step"] == 5500 and plan["register_kinds"] == []
    plan = cli.plan(["train", "--arm", "pfat0pdz_rep_cos", "--phase", "formal", "--config", str(R.config_path("pfat0pdz_rep_cos", "formal"))])
    assert plan["gpu_pair"] and plan["sampler"] == "dedicated" and plan["schedule"] == "cosine" and plan["round"] == 14
    plan = cli.plan(["train", "--arm", "pfagtdz", "--phase", "preflight", "--config", str(R.config_path("pfagtdz", "preflight"))])
    assert plan["family"] == "oracle" and "sampler" not in plan
    plan = cli.plan(["router-read", "d1", "--router", "srdwkda"])
    assert plan["kind"] == "dw" and plan["needs_gpu"]
    plan = cli.plan(["accept", "--arm", "pfat0pdz_replay", "--phase", "preflight"])
    assert plan["family"] == "replay" and not plan["needs_gpu"]
    with pytest.raises(RuntimeError, match="not registered"):
        cli.plan(["train", "--arm", "pfat0pdz_replay", "--phase", "formal", "--config", "x"])
    with pytest.raises(RuntimeError, match="Unregistered config path"):
        cli.plan(["train", "--arm", "pfat0pdz", "--phase", "formal", "--config", str(R.config_path("pfat0pdz", "preflight"))])
    with pytest.raises(RuntimeError, match="unregistered"):
        cli.plan(["accept", "--arm", "no_such_arm", "--phase", "formal"])


def test_cpu_verbs_run_in_process(capsys):
    assert cli.main(["configs", "--check"]) == 0
    assert "configs equal their generator" in capsys.readouterr().out
    assert cli.main(["verify-protocol"]) == 0
    assert "OK" in capsys.readouterr().out
    assert cli.main(["jobs", "standard", "--round", "13", "--dry-run"]) == 0
    assert "sm:r13:formal:pfat0pdz_rep" in capsys.readouterr().out


def test_select_needs_its_router_argument():
    with pytest.raises(RuntimeError, match="needs --router"):
        cli.main(["select", "--id", "r6c"])
    with pytest.raises(RuntimeError, match="r6c only"):
        cli.main(["select", "--id", "r4", "--router", "sr1"])


def run_cli(*args, module=False):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-W", "ignore"] + (["-m", f"{PKG}.cli"] if module else [str(CLI_FILE)]) + list(args)
    done = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300)
    return done.returncode, done.stdout


def test_the_script_and_the_module_start_identically():
    argv = ("plan", "evaluate", "--net", "xt0_srdwa", "--mode", "e1", "--step", "5500")
    code_a, out_a = run_cli(*argv)
    code_b, out_b = run_cli(*argv, module=True)
    assert code_a == code_b == 0 and json.loads(out_a) == json.loads(out_b) and json.loads(out_a)["trunk"] == "pfat0pdz"
    code, out = run_cli("verify-protocol")
    assert code == 0 and "code identity" in out


def test_the_script_does_not_shadow_the_standard_library():
    """tools/s45sm holds train.py, evaluate.py, ...: running the script must not put this directory on sys.path."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    code = ("import runpy, sys; sys.argv = [%r, 'plan', 'replay', '--arm', 'pfat0pdz_replay']; "
            "import json, io, contextlib; buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    try:\n        runpy.run_path(%r, run_name='__main__')\n    except SystemExit as e:\n        assert e.code == 0\n"
            "import pathlib; here = pathlib.Path(%r).resolve().parent\n"
            "assert all(pathlib.Path(p or '.').resolve() != here for p in sys.path)\n"
            "assert json.loads(buf.getvalue())['verb'] == 'replay'\nprint('ok')") % (str(CLI_FILE), str(CLI_FILE), str(CLI_FILE))
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-500:]
