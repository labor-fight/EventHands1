"""Opt-in, isolated FP32 CUDA graphs for S37 FK and active-head decoding.

Create ``CapturedS37Decode(model)`` under explicit ``torch.no_grad()`` after
loading/freezing/eval of the final model on the current CUDA device. Preparation
warms and captures two fixed-shape primitives. ``with context`` installs only
that model instance's _fk/_decode_active and restores them on exit; install()
and restore() are also available. Every call copies current inputs and clones
outputs, so successive calls cannot overwrite retained results.

The original MANO forward is inspected and exactly one AST expression is
replaced: transform_chain[self.parents[i]] uses an equivalent host parent tuple
to avoid capture-time scalar synchronization. No other AST operation is changed.
MANO.forward is overridden only during FK preparation and restored in finally.

This captures neither the encoder, routing, full forward, state update nor the
streaming service. Setup/guard/copy/clone costs remain; no 7 ms claim follows.
One isolated model/thread/device/stream is required. Version guards cannot
detect writes through .data, raw storage, or external CUDA code: such mutation
and uncoordinated concurrent use are outside this explicit ownership contract.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap
import threading
import time
import types

import torch
from torch import nn


def _method_identity(value):
    return (id(getattr(value, "__self__", None)), id(getattr(value, "__func__", value)))


def _compile_host_parent_forward(function):
    """Inspect/compile definitions only; prove one reversible AST substitution."""
    if not inspect.isfunction(function) or function.__name__ != "forward":
        raise TypeError("expected the original Python ManoLayer.forward function")
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("MANO source must contain exactly one forward definition")
    original = tree.body[0]
    if original.decorator_list or function.__closure__:
        raise ValueError("decorated/closure-based MANO forward is unsupported")
    if any(isinstance(node, ast.Name) and node.id == "_captured_mano_parents" for node in ast.walk(original)):
        raise ValueError("reserved host-parent name already occurs in original MANO source")
    expected = ast.parse("transform_chain[self.parents[i]]", mode="eval").body
    replacement = ast.parse("transform_chain[_captured_mano_parents[i]]", mode="eval").body

    class Replace(ast.NodeTransformer):
        def __init__(self, match, value):
            self.match = ast.dump(match, include_attributes=False)
            self.value, self.count = value, 0

        def visit_Subscript(self, node):
            if ast.dump(node, include_attributes=False) == self.match:
                self.count += 1
                return ast.copy_location(copy.deepcopy(self.value), node)
            return self.generic_visit(node)

    change = Replace(expected, replacement)
    adapted = change.visit(copy.deepcopy(original))
    if change.count != 1:
        raise ValueError(f"expected exactly one MANO parent index replacement, got {change.count}")
    undo = Replace(replacement, expected)
    recovered = undo.visit(copy.deepcopy(adapted))
    original_dump = ast.dump(original, include_attributes=False)
    if undo.count != 1 or ast.dump(recovered, include_attributes=False) != original_dump:
        raise RuntimeError("MANO AST reverse check failed")
    module = ast.fix_missing_locations(ast.Module(body=[adapted], type_ignores=[]))
    filename = (inspect.getsourcefile(function) or "<ManoLayer.forward>") + "::captured_host_parent"
    code = compile(module, filename, "exec", dont_inherit=True)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return code, {
        "replacement_count": change.count,
        "replacement": "transform_chain[self.parents[i]] -> transform_chain[_captured_mano_parents[i]]",
        "all_other_ast_nodes_preserved": True,
        "original_source_sha256": digest(source),
        "original_ast_sha256": digest(original_dump),
        "adapted_ast_sha256": digest(ast.dump(adapted, include_attributes=False)),
    }


class CapturedS37Decode:
    """Prepared fixed-shape graphs owned by one frozen S37 eval instance.

    Input contracts: FK pose(1,51), betas(1,10); active decode feat(1,512),
    prev(1,51), evidence(1,16,257). No fallback, mixed precision or root_extra.
    Current betas/pose/features/evidence are copied on EVERY invocation.
    Model context K/betas and route_stats may change outside captured methods:
    they are not captured state; FK receives betas/pose explicitly each time.
    """
    _REQUIRED_TRUE = ("routed", "active_head", "predict_delta", "prevpos_embed", "zero_event_gate")
    _REQUIRED_FALSE = ("prev_render", "root_covmap", "fk_graph", "mesh_graph", "mesh_query",
                       "event_guided_mesh", "egm_memory", "ablate_evidence", "ablate_joint_heads")
    _HOOK_NAMES = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")

    def __init__(self, model):
        started = time.perf_counter_ns()
        self.model = model
        first = next(model.parameters(), None)
        if first is None or first.device.type != "cuda":
            raise ValueError("CapturedS37Decode requires a CUDA model")
        self.device = first.device
        self.owner_thread = threading.get_ident()
        self.owner_stream = torch.cuda.current_stream(self.device).cuda_stream
        self._installed = False
        self._busy = False
        self._original_fk = model._fk
        self._original_decode = model._decode_active
        self._original_mano_forward = model.mano.forward
        for owner, name in ((model, "_fk"), (model, "_decode_active"), (model.mano, "forward")):
            if name in owner.__dict__:
                raise ValueError(f"existing instance override of {name} is unsupported")
        self._check_supported_model()
        self._frozen_state = self._state_signature()
        self._frozen_modules = self._module_signature()
        self._frozen_flags = self._flags()
        self._frozen_math = self._math_flags()
        self._dependencies = self._function_dependencies()
        self._frozen_dependencies = self._dependency_signature()
        self._validate_runtime()

        code, ast_metadata = _compile_host_parent_forward(self._original_mano_forward.__func__)
        parents = tuple(int(value) for value in model.mano.parents.detach().cpu().tolist())
        if len(parents) != 16 or parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, 16)):
            raise ValueError("unsupported MANO parent topology")
        namespace = dict(self._original_mano_forward.__func__.__globals__)
        namespace["_captured_mano_parents"] = parents
        exec(code, namespace)
        adapted = types.MethodType(namespace["forward"], model.mano)
        self.metadata = {
            "scope": "FK and active-head CUDA schedules only; not a captured full model or latency acceptance",
            "parent_indices": parents, "ast": ast_metadata,
            "owner_thread": self.owner_thread, "owner_cuda_stream": self.owner_stream,
            "device": str(self.device), "torch_version": torch.__version__,
            "input_contract": {"fk": [[1, 51], [1, 10]], "decode": [[1, 512], [1, 51], [1, 16, 257]]},
            "output_ownership": "copy current inputs, replay, clone every output",
            "stages": {},
            "primitive_source_sha256": {
                name: hashlib.sha256(inspect.getsource(method.__func__).encode()).hexdigest()
                for name, method in (("fk", self._original_fk), ("decode_active", self._original_decode),
                                     ("mano_forward", self._original_mano_forward))},
        }
        pose = torch.zeros(1, 51, device=self.device, dtype=torch.float32)
        pose[:, 2] = .45
        self._fk_inputs = (pose, torch.zeros(1, 10, device=self.device, dtype=torch.float32))
        self._decode_inputs = (torch.zeros(1, 512, device=self.device, dtype=torch.float32), pose.clone(),
                               torch.zeros(1, 16, 257, device=self.device, dtype=torch.float32))
        # No persistent MANO patch, including if warmup/capture raises.
        model.mano.forward = adapted
        try:
            self._fk_graph, self._fk_output = self._capture("fk", self._original_fk, self._fk_inputs)
        finally:
            delattr(model.mano, "forward")
        self._decode_graph, self._decode_output = self._capture(
            "decode_active", self._original_decode, self._decode_inputs)
        self._validate_runtime()
        self.metadata["setup_total_ms"] = (time.perf_counter_ns() - started) / 1e6

    @property
    def original_fk(self):
        return self._original_fk

    @property
    def original_decode(self):
        return self._original_decode

    def _check_supported_model(self):
        m = self.model
        if any(getattr(m, name, None) is not True for name in self._REQUIRED_TRUE):
            raise ValueError("requires the frozen routed S37 active/delta/prev/empty-gate interface")
        if any(bool(getattr(m, name, False)) for name in self._REQUIRED_FALSE):
            raise ValueError("unsupported S37 branch or ablation")
        if (m.encoder_name, m.pose_repr, m.output_dim) != ("event_gnn", "mano_full_axis_angle", 51):
            raise ValueError("requires event_gnn and full 51D axis-angle pose")
        if getattr(m, "route_prev_override", None) is not None:
            raise ValueError("oracle/alternate routing state is unsupported")
        if m.mano.add_mean or m.mano.num_joints != 16:
            raise ValueError("requires original 16-joint MANO add_mean=False")
        if not isinstance(m.root_head, nn.Linear) or (m.root_head.in_features, m.root_head.out_features) != (4624, 6):
            raise ValueError("requires original feat512/evidence16x257 root head")
        if not isinstance(m.joint_heads, nn.ModuleList) or len(m.joint_heads) != 15:
            raise ValueError("requires fifteen original joint heads")
        for head in m.joint_heads:
            if (not isinstance(head, nn.Sequential) or len(head) != 3 or
                    not isinstance(head[0], nn.Linear) or not isinstance(head[1], nn.ReLU) or
                    not isinstance(head[2], nn.Linear) or head[0].in_features != 260 or head[2].out_features != 3):
                raise ValueError("unsupported joint-head topology")
        if (m.event_encoder.hidden, m.event_encoder.feat_dim) != (128, 512):
            raise ValueError("requires original hidden128/feat512 event encoder")

    def _state_signature(self):
        rows = []
        for kind, items in (("parameter", self.model.named_parameters()), ("buffer", self.model.named_buffers())):
            for name, value in items:
                if value.device != self.device or (value.is_floating_point() and value.dtype != torch.float32):
                    raise RuntimeError("all model state must share the context CUDA device and floating FP32 dtype")
                if value.requires_grad:
                    raise RuntimeError("all model parameters/buffers must be frozen")
                try:
                    version = value._version
                except RuntimeError as error:
                    raise RuntimeError("model state must expose version counters; construct outside inference_mode") from error
                rows.append((kind, name, id(value), version, value.data_ptr(), value.dtype,
                             tuple(value.shape), tuple(value.stride()), value.storage_offset()))
        return tuple(rows)

    def _module_signature(self):
        rows = []
        for name, module in self.model.named_modules():
            if module.training:
                raise RuntimeError("all model modules must remain in eval mode")
            if any(getattr(module, key, None) for key in self._HOOK_NAMES):
                raise RuntimeError("module forward/backward hooks are unsupported during graph ownership")
            # Captured Linear/ReLU and MANO public scalar attributes affect
            # behavior; live route_stats/_ctx_K/_ctx_betas are deliberately not
            # captured dependencies. Model branch flags are checked separately.
            scalars = tuple(sorted((key, value) for key, value in vars(module).items()
                                   if not key.startswith("_") and isinstance(value, (bool, int, float, str, type(None)))))
            rows.append((name, id(module), type(module), _method_identity(module.forward), scalars))
        return tuple(rows)

    def _flags(self):
        m = self.model
        names = self._REQUIRED_TRUE + self._REQUIRED_FALSE + (
            "encoder_name", "pose_repr", "output_dim", "route_band_px", "route_front_k")
        if getattr(m, "route_prev_override", None) is not None:
            raise RuntimeError("routing override changed during graph ownership")
        return tuple((name, getattr(m, name, None)) for name in names)

    @staticmethod
    def _math_flags():
        return (torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic)

    def _function_dependencies(self):
        # Record project helper globals actually named by the captured Python
        # functions; frozen CUDA graphs must not silently outlive code changes.
        pending = [self._original_fk.__func__, self._original_decode.__func__, self._original_mano_forward.__func__]
        seen, result = set(), []
        while pending:
            function = pending.pop()
            if id(function) in seen:
                continue
            seen.add(id(function))
            result.append((function, None))
            for name in function.__code__.co_names:
                if name not in function.__globals__:
                    continue
                value = function.__globals__[name]
                result.append((function, name))
                if inspect.isfunction(value) and value.__module__ == function.__module__:
                    pending.append(value)
                elif inspect.isfunction(value) and name == "decode_to_mano_inputs":
                    pending.append(value)
        return tuple(result)

    def _dependency_signature(self):
        result = []
        for function, name in self._dependencies:
            value = function if name is None else function.__globals__.get(name)
            result.append((id(value), id(getattr(value, "__code__", None))))
        return tuple(result)

    def _validate_runtime(self):
        # Cross-thread rejection occurs before CUDA queries or input mutation.
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("captured decode called from another thread")
        if torch.is_grad_enabled():
            raise RuntimeError("captured decode requires explicit no_grad")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("captured decode forbids inference_mode")
        if (not torch.are_deterministic_algorithms_enabled() or
                torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError("captured decode requires strict deterministic algorithms")
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise RuntimeError("captured decode forbids autocast")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("captured decode called with another current CUDA device")
        if torch.cuda.current_stream(self.device).cuda_stream != self.owner_stream:
            raise RuntimeError("captured decode called from another CUDA stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("nested/outer CUDA stream capture is unsupported")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("captured decode requires TF32 disabled")
        if torch.get_float32_matmul_precision() != "highest" or self._math_flags() != self._frozen_math:
            raise RuntimeError("floating-point/determinism policy changed")
        global_hooks = torch.nn.modules.module
        if any(getattr(global_hooks, "_global" + name, None) for name in self._HOOK_NAMES):
            raise RuntimeError("global module hooks are unsupported during graph ownership")
        if self._state_signature() != self._frozen_state:
            raise RuntimeError("model parameters/buffers changed; recreate capture")
        if self._module_signature() != self._frozen_modules or self._flags() != self._frozen_flags:
            raise RuntimeError("model modules/methods/flags changed; recreate capture")
        if self._dependency_signature() != self._frozen_dependencies:
            raise RuntimeError("captured Python helpers changed; recreate capture")
        expected_fk = self.fk if self._installed else self._original_fk
        expected_decode = self.decode_active if self._installed else self._original_decode
        if (_method_identity(self.model._fk) != _method_identity(expected_fk) or
                _method_identity(self.model._decode_active) != _method_identity(expected_decode)):
            raise RuntimeError("isolated model method ownership changed")

    def _inputs(self, values, shapes):
        self._validate_runtime()
        if self._busy:
            raise RuntimeError("reentrant captured decode call is unsupported")
        for value, shape in zip(values, shapes):
            if not isinstance(value, torch.Tensor):
                raise TypeError("captured inputs must be tensors")
            if tuple(value.shape) != shape or value.dtype != torch.float32 or value.device != self.device:
                raise ValueError(f"captured input must be FP32 shape {shape} on {self.device}")
            if value.requires_grad or value.layout != torch.strided:
                raise ValueError("captured inputs must be ordinary frozen strided tensors")

    def _capture(self, name, function, inputs):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        tick = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(*inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        warmup_ms = (time.perf_counter_ns() - tick) / 1e6
        tick = time.perf_counter_ns()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = function(*inputs)
        torch.cuda.synchronize(self.device)
        self.metadata["stages"][name] = {
            "prerequisite_warmup_ms": warmup_ms, "capture_ms": (time.perf_counter_ns() - tick) / 1e6}
        return graph, output

    def fk(self, params, betas):
        self._inputs((params, betas), ((1, 51), (1, 10)))
        self._busy = True
        try:
            for destination, source in zip(self._fk_inputs, (params, betas)):
                destination.copy_(source)
            self._fk_graph.replay()
            return tuple(value.clone() for value in self._fk_output)
        finally:
            self._busy = False

    def decode_active(self, feat, prevpos, evidence=None, root_extra=None):
        if evidence is None or root_extra is not None:
            raise ValueError("captured decode requires evidence and forbids root_extra")
        self._inputs((feat, prevpos, evidence), ((1, 512), (1, 51), (1, 16, 257)))
        self._busy = True
        try:
            for destination, source in zip(self._decode_inputs, (feat, prevpos, evidence)):
                destination.copy_(source)
            self._decode_graph.replay()
            return self._decode_output.clone()
        finally:
            self._busy = False

    def install(self):
        self._validate_runtime()
        if self._installed:
            raise RuntimeError("captured wrappers are already installed")
        self._old_overrides = {name: (name in self.model.__dict__, self.model.__dict__.get(name))
                               for name in ("_fk", "_decode_active")}
        self.model._fk = self.fk
        try:
            self.model._decode_active = self.decode_active
        except BaseException:
            existed, value = self._old_overrides["_fk"]
            if existed:
                self.model._fk = value
            else:
                delattr(self.model, "_fk")
            raise
        self._installed = True
        return self

    def restore(self):
        """Idempotent cleanup despite changed weights/flags; never clobber others."""
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("restore must run on the owning thread")
        if not self._installed:
            return
        conflicts = []
        for name, owned in (("_fk", self.fk), ("_decode_active", self.decode_active)):
            if _method_identity(getattr(self.model, name, None)) != _method_identity(owned):
                conflicts.append(name)
                continue
            existed, value = self._old_overrides[name]
            if existed:
                setattr(self.model, name, value)
            else:
                delattr(self.model, name)
        self._installed = False
        if conflicts:
            raise RuntimeError("foreign method replacements retained during restore: " + ", ".join(conflicts))

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False
