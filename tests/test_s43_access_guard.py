"""Regression tests for the process-wide, bounded inference audit guard."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tools.s43_rel import access_guard as G


ROOT = Path(__file__).resolve().parents[1]


def run_isolated(script):
    """Permanent Python audit hooks must be tested in disposable processes."""
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_thousand_scopes_register_only_one_real_audit_hook():
    run_isolated("""
        import sys
        from tools.s43_rel import access_guard as G

        original = sys.addaudithook
        registrations = []
        def register(hook):
            registrations.append(hook)
            return original(hook)
        sys.addaudithook = register

        for _ in range(1000):
            with G.InferenceAccessGuard() as guard:
                sys.audit('open', 'allowed.bin', 'r', 0)
            assert not guard.active
            assert G._active_guards == ()
        assert G._audit_hook_registered
        assert len(registrations) == 1, len(registrations)
        assert registrations[0] is G._audit_open
    """)


def test_completed_scope_does_not_keep_guard_alive():
    guard = G.InferenceAccessGuard()
    reference = weakref.ref(guard)
    with guard:
        assert guard.active
    del guard
    gc.collect()
    assert reference() is None
    assert G._active_guards == ()


def test_real_metadata_and_beta_access_are_blocked_and_exit_restores(tmp_path):
    metadata = tmp_path / 'sample.meta'
    metadata.write_bytes(b'score-only pose')
    archive = tmp_path / 'sample_aux.npz'
    np.savez(archive, camera_K=np.eye(3), betas=np.arange(10))
    original = np.lib.npyio.NpzFile.__getitem__

    with G.InferenceAccessGuard() as guard:
        with pytest.raises(RuntimeError, match=r'pose \.meta'):
            metadata.read_bytes()
        with np.load(archive) as values:
            np.testing.assert_array_equal(values['camera_K'], np.eye(3))
            with pytest.raises(RuntimeError, match='betas'):
                values['betas']

    assert np.lib.npyio.NpzFile.__getitem__ is original
    assert guard.receipt() == {
        'active': False,
        'aux_betas_accessed': False,
        'pose_metadata_opened': False,
        'blocked_aux_beta_attempts': 1,
        'blocked_pose_meta_attempts': 1,
        'aux_members_read': [{'path': str(archive), 'member': 'camera_K'}],
    }
    assert metadata.read_bytes() == b'score-only pose'
    with np.load(archive) as values:
        np.testing.assert_array_equal(values['betas'], np.arange(10))


def test_nested_scope_exit_keeps_outer_firewall_and_restores_after_exception(tmp_path):
    metadata = tmp_path / 'sample.meta'
    metadata.write_bytes(b'pose')
    original = np.lib.npyio.NpzFile.__getitem__

    with pytest.raises(ValueError, match='body failure'):
        with G.InferenceAccessGuard() as outer:
            with G.InferenceAccessGuard() as inner:
                with pytest.raises(RuntimeError, match=r'pose \.meta'):
                    metadata.read_bytes()
            assert outer.active and not inner.active
            with pytest.raises(RuntimeError, match=r'pose \.meta'):
                metadata.read_bytes()
            raise ValueError('body failure')

    assert not outer.active and not inner.active
    assert G._active_guards == ()
    assert np.lib.npyio.NpzFile.__getitem__ is original
    assert metadata.read_bytes() == b'pose'


def test_same_guard_cannot_be_entered_twice(tmp_path):
    metadata = tmp_path / 'sample.meta'
    metadata.write_bytes(b'pose')
    guard = G.InferenceAccessGuard()
    with guard:
        with pytest.raises(RuntimeError, match='active'):
            guard.__enter__()
        assert guard.active
        with pytest.raises(RuntimeError, match=r'pose \.meta'):
            metadata.read_bytes()
    assert not guard.active
    assert metadata.read_bytes() == b'pose'


def test_worker_thread_obeys_active_process_firewall(tmp_path):
    metadata = tmp_path / 'sample.meta'
    metadata.write_bytes(b'pose')
    archive = tmp_path / 'sample_aux.npz'
    np.savez(archive, betas=np.arange(10))

    def read_beta():
        with np.load(archive) as values:
            return values['betas']

    with ThreadPoolExecutor(max_workers=1) as pool:
        with G.InferenceAccessGuard() as guard:
            with pytest.raises(RuntimeError, match=r'pose \.meta'):
                pool.submit(metadata.read_bytes).result(timeout=5)
            with pytest.raises(RuntimeError, match='betas'):
                pool.submit(read_beta).result(timeout=5)
        assert pool.submit(metadata.read_bytes).result(timeout=5) == b'pose'

    receipt = guard.receipt()
    assert receipt['blocked_pose_meta_attempts'] == 1
    assert receipt['blocked_aux_beta_attempts'] == 1


def test_overlapping_thread_scopes_restore_numpy_only_after_last_exit(tmp_path):
    archive = tmp_path / 'sample_aux.npz'
    np.savez(archive, betas=np.arange(10))
    entered = threading.Event()
    finish = threading.Event()
    original = np.lib.npyio.NpzFile.__getitem__

    def worker():
        with G.InferenceAccessGuard():
            entered.set()
            assert finish.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        try:
            assert entered.wait(timeout=5)
            with G.InferenceAccessGuard():
                finish.set()
                future.result(timeout=5)
                with np.load(archive) as values:
                    with pytest.raises(RuntimeError, match='betas'):
                        values['betas']
        finally:
            finish.set()
        future.result(timeout=5)

    assert G._active_guards == ()
    assert np.lib.npyio.NpzFile.__getitem__ is original


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires POSIX fork')
def test_fork_reuses_inherited_hook_without_inheriting_active_scopes():
    run_isolated("""
        import json
        import os
        import sys
        import numpy as np
        from tools.s43_rel import access_guard as G

        original_add = sys.addaudithook
        registrations = []
        def register(hook):
            registrations.append(hook)
            return original_add(hook)
        sys.addaudithook = register
        original_getitem = np.lib.npyio.NpzFile.__getitem__
        read_fd, write_fd = os.pipe()
        with G.InferenceAccessGuard() as parent_guard:
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    assert G._active_guards == ()
                    assert np.lib.npyio.NpzFile.__getitem__ is original_getitem
                    sys.audit('open', 'allowed-in-child.meta', 'r', 0)
                    with G.InferenceAccessGuard() as child_guard:
                        try:
                            sys.audit('open', 'blocked-in-child.meta', 'r', 0)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError('child guard failed to block metadata')
                    assert child_guard.receipt()['blocked_pose_meta_attempts'] == 1
                    assert len(registrations) == 1, len(registrations)
                    result = {'ok': True}
                except BaseException as error:
                    result = {'ok': False, 'error': repr(error)}
                os.write(write_fd, json.dumps(result).encode())
                os.close(write_fd)
                os._exit(0 if result['ok'] else 1)
            os.close(write_fd)
            payload = os.read(read_fd, 65536)
            os.close(read_fd)
            _, status = os.waitpid(pid, 0)
            assert status == 0, payload
            assert json.loads(payload)['ok']
            assert parent_guard.active
            try:
                sys.audit('open', 'still-blocked-in-parent.meta', 'r', 0)
            except RuntimeError:
                pass
            else:
                raise AssertionError('fork changed parent firewall')
        assert np.lib.npyio.NpzFile.__getitem__ is original_getitem
        assert len(registrations) == 1
    """)
