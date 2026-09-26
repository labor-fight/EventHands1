"""Fixed-order CUDA FP32 linear primitive for exact event-prefix execution.

No parameters, autotuning, TF32, or batch-size-dependent reduction schedule.
The backward uses matrix products, not an expanded per-row weight gradient.
CPU execution and the legacy S37/C0 modules keep their original nn.Linear.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['rows'])
def _linear_kernel(X, W, B, Y, rows, IN: tl.constexpr, OUT: tl.constexpr):
    r = tl.program_id(0) * 16 + tl.arange(0, 16)
    c = tl.program_id(1) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 32)
    acc = tl.zeros((16, 32), dtype=tl.float32)
    for block in range(tl.cdiv(IN, 32)):
        kk = block * 32 + k
        x = tl.load(X + r[:, None] * IN + kk[None, :],
                    (r[:, None] < rows) & (kk[None, :] < IN), other=0.)
        w = tl.load(W + c[None, :] * IN + kk[:, None],
                    (c[None, :] < OUT) & (kk[:, None] < IN), other=0.)
        acc += tl.dot(x, w, allow_tf32=False)
    bias = tl.load(B + c, c < OUT, other=0.)
    tl.store(Y + r[:, None] * OUT + c[None, :], acc + bias[None, :],
             (r[:, None] < rows) & (c[None, :] < OUT))


class _FixedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if (not x.is_cuda or x.dtype != torch.float32 or weight.dtype != torch.float32
                or bias.dtype != torch.float32 or torch.is_autocast_enabled()):
            raise ValueError('fixed streaming linear requires CUDA FP32 without autocast')
        if x.device != weight.device or x.device != bias.device:
            raise ValueError('linear input and parameters must share one CUDA device')
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        if weight.shape != (bias.numel(), flat.shape[1]):
            raise ValueError('invalid linear dimensions')
        result = flat.new_empty((len(flat), len(bias)))
        if len(flat):
            _linear_kernel[(triton.cdiv(len(flat), 16), triton.cdiv(len(bias), 32))](
                flat, weight, bias, result, len(flat), flat.shape[1], len(bias),
                num_warps=4, num_stages=1)
        ctx.save_for_backward(flat, weight)
        ctx.input_shape = x.shape
        return result.reshape(*x.shape[:-1], len(bias))

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        g = grad_output.reshape(-1, weight.shape[0]).contiguous()
        gx = (g @ weight).reshape(ctx.input_shape) if ctx.needs_input_grad[0] else None
        gw = g.t() @ x if ctx.needs_input_grad[1] else None
        gb = g.sum(0) if ctx.needs_input_grad[2] else None
        return gx, gw, gb


def fixed_linear(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
    """Same affine map and parameter objects; only the CUDA evaluation order differs."""
    if layer.bias is None:
        raise ValueError('the S37 streaming contract requires a bias')
    return _FixedLinear.apply(x, layer.weight, layer.bias)
