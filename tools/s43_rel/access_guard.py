"""Process-wide inference firewall with one permanent Python audit hook.

This is the opt-in replacement for the guard in frozen mean_shape_replay.py.
Only active scopes are retained; ending a scope never leaves a bound-method
hook behind. The policy remains process-wide, including input worker threads.
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np


_lock = threading.RLock()
_audit_hook_registered = False
_audit_hook_registrations = 0
_active_guards = ()
_npz_class = None
_original_getitem = None


def _audit_open(event, args):
    if event != 'open' or not args:
        return
    guards = _active_guards
    if not guards:
        return
    path = args[0]
    if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).endswith('.meta'):
        # Python dispatches the original audit hooks in registration order.
        guard = guards[0]
        guard.blocked_pose_meta_attempts += 1
        raise RuntimeError('inference firewall forbids opening pose .meta')


def _ensure_audit_hook():
    global _audit_hook_registered, _audit_hook_registrations
    with _lock:
        if not _audit_hook_registered:
            sys.addaudithook(_audit_open)
            _audit_hook_registered = True
            _audit_hook_registrations += 1


def _guarded_getitem(archive, key):
    guards = _active_guards
    # Capture the original before another thread can end the final scope.
    original = _original_getitem
    if key == 'betas' and guards:
        # The original nested NumPy wrappers check the innermost scope first.
        guards[-1].blocked_aux_beta_attempts += 1
        raise RuntimeError('inference firewall forbids reading NPZ betas field')
    if guards:
        path = getattr(getattr(archive, 'zip', None), 'filename', None)
        if path is not None and str(path).endswith('_aux.npz'):
            for guard in guards:
                guard.aux_members_read.append({'path': str(path), 'member': str(key)})
    return original(archive, key)


def _after_fork():
    global _lock, _active_guards
    # A Python audit hook survives fork. Reuse it, but do not inherit scopes
    # or a lock held by a vanished thread from the parent process.
    _lock = threading.RLock()
    for guard in _active_guards:
        guard.active = False
    _active_guards = ()
    if _npz_class is not None and _npz_class.__getitem__ is _guarded_getitem:
        _npz_class.__getitem__ = _original_getitem


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def diagnostics():
    return {
        'implementation': 'single_process_audit_dispatcher_v1',
        'pid': os.getpid(),
        'audit_hook_registered': _audit_hook_registered,
        'audit_hook_registrations': _audit_hook_registrations,
        'active_guards': len(_active_guards),
    }


class InferenceAccessGuard:
    """Same inference access policy/receipt as the frozen guard, bounded hooks."""

    def __init__(self):
        self.active = False
        self.blocked_aux_beta_attempts = 0
        self.blocked_pose_meta_attempts = 0
        self.aux_members_read = []

    def __enter__(self):
        global _active_guards, _npz_class, _original_getitem
        with _lock:
            if self.active:
                raise RuntimeError('guard already active')
            _ensure_audit_hook()
            if not _active_guards:
                _npz_class = np.lib.npyio.NpzFile
                _original_getitem = _npz_class.__getitem__
                _npz_class.__getitem__ = _guarded_getitem
            self.active = True
            _active_guards = (*_active_guards, self)
        return self

    def __exit__(self, typ, value, traceback):
        global _active_guards
        with _lock:
            self.active = False
            _active_guards = tuple(guard for guard in _active_guards if guard is not self)
            if not _active_guards and _npz_class is not None:
                _npz_class.__getitem__ = _original_getitem
        return False

    def receipt(self):
        return {
            'active': self.active,
            'aux_betas_accessed': False,
            'pose_metadata_opened': False,
            'blocked_aux_beta_attempts': self.blocked_aux_beta_attempts,
            'blocked_pose_meta_attempts': self.blocked_pose_meta_attempts,
            'aux_members_read': list(self.aux_members_read),
        }
