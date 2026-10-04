"""Shared configuration overrides for the RT, S38 and DT recipes."""


def merge(dst: dict, src: dict) -> dict:
    """Merge overrides in place; a None override removes the destination key."""
    for k, v in src.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict):
            merge(dst.setdefault(k, {}), v)
        else:
            dst[k] = v
    return dst
