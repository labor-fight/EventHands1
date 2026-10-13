"""Read and update marked sections of the consolidated experiment records."""
from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path


class SectionSnapshot(str):
    """Section text with its original document, for a guarded later update."""

    def __new__(cls, text, *, path, key, source, fingerprint):
        snapshot = super().__new__(cls, text)
        snapshot.path = path
        snapshot.key = key
        snapshot.source = source
        snapshot.fingerprint = fingerprint
        return snapshot


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_document(path: Path):
    before = path.stat()
    source = path.read_bytes()
    after = path.stat()
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"document changed while reading: {path}")
    return source, after


def _section_bounds(text: str, key: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key):
        raise ValueError(f"invalid experiment section key: {key!r}")
    begin = f"<!-- EXPERIMENT_RECORD:{key}:BEGIN -->"
    end = f"<!-- EXPERIMENT_RECORD:{key}:END -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"section {key} must have exactly one BEGIN and one END marker")
    start, stop = text.index(begin) + len(begin), text.index(end)
    if stop < start:
        raise ValueError(f"section {key} END marker precedes its BEGIN marker")
    return start, stop


def get_section(path: str | Path, key: str) -> SectionSnapshot:
    """Return section text; pass this snapshot as ``expected`` when writing."""
    path = Path(path).resolve()
    source, info = _read_document(path)
    text = source.decode("utf-8")
    start, stop = _section_bounds(text, key)
    return SectionSnapshot(text[start:stop].strip("\r\n"), path=path, key=key,
                           source=source, fingerprint=_fingerprint(info))


def _markdown_nodes(text: str):
    """Locate explicit anchors and headings outside fenced code."""
    anchors, headings = [], []
    fence, position = None, 0
    for line in text.splitlines(keepends=True):
        plain = line.rstrip("\r\n")
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
        else:
            fence = _opens_fence(line)
            if fence is None:
                anchor = re.fullmatch(r' {0,3}<a id="([^"]+)"></a>[ \t]*', plain)
                heading = re.match(r"^ {0,3}#{1,6}(?=[ \t]|$)(.*)$", plain)
                if anchor:
                    anchors.append((anchor.group(1), position))
                if heading:
                    headings.append((heading.group(1).strip(), position))
        position += len(line)
    return anchors, headings


def _preserve_anchors(previous: str, content: str, key: str) -> str:
    """Keep chapter links valid when a producer replaces generated headings."""
    old_anchors, old_headings = _markdown_nodes(previous)
    new_anchors, new_headings = _markdown_nodes(content)
    present = {name for name, _ in new_anchors}
    if len(present) != len(new_anchors):
        raise ValueError("replacement contains duplicate explicit anchors")
    positions = {}
    for title, position in new_headings:
        positions.setdefault(title, position)
    main_position = next(position for name, position in new_anchors if name == key.lower())
    end_of_anchor = content.find("\n", main_position)
    fallback = end_of_anchor + 1 if end_of_anchor >= 0 else len(content)
    missing = {}
    for name, old_position in old_anchors:
        if name in present:
            continue
        title = next((title for title, position in old_headings if position > old_position), None)
        # Retain unmatched historical aliases at the chapter start rather than
        # dropping an existing external or relative link to this chapter.
        position = positions.get(title, fallback)
        missing.setdefault(position, []).append(f'<a id="{name}"></a>')
        present.add(name)
    for position, anchors in sorted(missing.items(), reverse=True):
        content = content[:position] + "\n".join(anchors) + "\n\n" + content[position:]
    return content


def replace_section(path: str | Path, key: str, content: str, *,
                    expected: SectionSnapshot | None = None) -> None:
    """Atomically replace one section, refusing an outdated document snapshot.

    Producers should call ``get_section`` before generating their output and
    supply that result as ``expected``. With no snapshot, the current document
    is read immediately before constructing the replacement.
    """
    path = Path(path).resolve()
    if expected is None:
        expected = get_section(path, key)
    if not isinstance(expected, SectionSnapshot):
        raise TypeError("expected must be the snapshot returned by get_section")
    if expected.path != path or expected.key != key:
        raise ValueError("snapshot belongs to a different document or section")
    source, info = _read_document(path)
    if source != expected.source or _fingerprint(info) != expected.fingerprint:
        raise RuntimeError(f"document changed since section {key} was read: {path}")
    text = source.decode("utf-8")
    start, stop = _section_bounds(text, key)
    anchor = f'<a id="{key.lower()}"></a>'
    content_anchors, _ = _markdown_nodes(content)
    if key.lower() not in {name for name, _ in content_anchors}:
        content = anchor + "\n\n" + content
    content = _preserve_anchors(text[start:stop], content, key)
    replacement = text[:start] + "\n" + content.strip("\r\n") + "\n" + text[stop:]
    _section_bounds(replacement, key)
    replacement_anchors, _ = _markdown_nodes(replacement)
    if sum(name == key.lower() for name, _ in replacement_anchors) != 1:
        raise ValueError(f"section {key} must have exactly one anchor {key.lower()}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            os.fchmod(output.fileno(), stat.S_IMODE(info.st_mode))
            output.write(replacement.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        current, current_info = _read_document(path)
        if current != source or _fingerprint(current_info) != expected.fingerprint:
            raise RuntimeError(f"document changed before section {key} could be written: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _opens_fence(line: str):
    opening = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
    if opening:
        marker = opening.group(1)
        return marker[0], len(marker)
    return None


def _closes_fence(line: str, fence) -> bool:
    return re.fullmatch(r" {0,3}" + re.escape(fence[0]) + r"{" + str(fence[1]) + r",}[ \t]*",
                        line.rstrip("\r\n")) is not None


def demote_headings(text: str, offset: int = 2) -> str:
    """Demote ATX headings without changing fenced code or other Markdown."""
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("heading offset must be a nonnegative integer")
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
            lines.append(line)
            continue
        fence = _opens_fence(line)
        if fence is None:
            heading = re.match(r"^( {0,3})(#{1,6})(?=[ \t]|\r?$)", line)
            if heading:
                depth = min(6, len(heading.group(2)) + offset)
                line = heading.group(1) + "#" * depth + line[heading.end():]
        lines.append(line)
    return "".join(lines)
