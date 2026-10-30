"""Temporary, frozen-weight XCL interventions; all hooks are restored on exit."""
from __future__ import annotations

from contextlib import contextmanager

NAMES = ("none", "geometry_zero", "h3_zero", "residual_zero", "route_a2", "route_a3")


@contextmanager
def apply(model, name="none"):
    if name not in NAMES:
        raise ValueError(f"Unknown intervention {name!r}; choose {NAMES}")
    if name == "none":
        yield
        return
    xyz = model.xyz
    hooks = []
    saved_transform = getattr(xyz, "xag_h3_transform", None)
    saved_assoc = getattr(model, "xag_assoc_mode", "base")
    try:
        if name == "geometry_zero":
            # token7 | dXYZ3 | zn1 | valid1 | hit1; leave edge geometry intact.
            def relation_input(module, inputs):
                value = inputs[0].clone()
                value[..., 7:11] = 0
                return (value, *inputs[1:])
            hooks.append(xyz.rel[0].register_forward_pre_hook(relation_input))
        elif name == "h3_zero":
            xyz.xag_h3_transform = lambda h, s, o: h * 0
        elif name == "residual_zero":
            # The final group is the residual slot in the common 17-slot readout.
            def slot_input(module, inputs):
                value = inputs[0].clone()
                if module.groups != 17 or value.shape[1] % 17:
                    raise ValueError("Residual intervention requires the original 17-slot readout")
                value[:, -value.shape[1] // 17:] = 0
                return (value, *inputs[1:])
            hooks.append(xyz.slot_proj.register_forward_pre_hook(slot_input))
        else:
            model.xag_assoc_mode = "face_route" if name == "route_a2" else "full_oracle"
        yield
    finally:
        for hook in hooks:
            hook.remove()
        xyz.xag_h3_transform = saved_transform
        model.xag_assoc_mode = saved_assoc


def adapt_config(cfg, name):
    """Clone evaluation data settings only; never write a training configuration."""
    import copy
    from pathlib import Path
    cfg = copy.deepcopy(cfg)
    if name.startswith("route_"):
        root = Path(__file__).resolve().parents[2]
        cfg.setdefault("DATA", {})["XAG_SOURCE_ROOT"] = str(root / "outputs/xag/sources")
        cfg.setdefault("MODEL", {})["XAG_ASSOC"] = "face_route" if name == "route_a2" else "full_oracle"
    return cfg
