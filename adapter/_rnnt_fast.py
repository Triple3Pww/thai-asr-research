#!/usr/bin/env python3
"""A vectorised RNN-T loss in plain PyTorch, for a venv where the fast ones do not run.

    python adapter/_rnnt_fast.py          # selftest: matches NeMo's reference loss, and times both

WHY THIS EXISTS (2026-10-03). Fine-tuning typhoon-asr-realtime needs an RNN-T loss. In
E:/asr-spike/.venv:
  * NeMo's default `warprnnt_numba` dies on step 1 under numba 0.67 ("Signature mismatch").
  * NeMo's `pytorch` loss is correct but loops in Python over every (t, u) cell: 100 s per batch
    of 16 x 20 s clips -- 16 hours an epoch.
  * torchaudio and k2 are not installed, and CLAUDE.md forbids perturbing this venv.

The forward variable alpha(t, u) obeys
    alpha(t, u) = logaddexp( alpha(t-1, u) + blank(t-1, u),  alpha(t, u-1) + emit(t, u-1) )
Within one t the u-recursion is linear in log space, so it closes in one `logcumsumexp`:
    alpha(t, u) = C(u) + logcumsumexp_k( a(k) - C(k) ),   a(k) = alpha(t-1,k)+blank(t-1,k),
                                                          C(u) = sum_{j<u} emit(t, j)
That is one Python iteration per frame (~250 for 20 s at 80 ms) instead of one per cell, and
autograd provides the backward pass. The loss is -[alpha(T-1, U) + blank(T-1, U)] per utterance.
"""

from __future__ import annotations

import torch
from torch import nn


def rnnt_nll(acts: torch.Tensor, labels: torch.Tensor, act_lens: torch.Tensor,
             label_lens: torch.Tensor, blank: int) -> torch.Tensor:
    """acts: [B, T, U+1, V] logits; labels: [B, U]; returns per-utterance NLL [B]."""
    lp = torch.log_softmax(acts.float(), dim=-1)
    B, T, U1, _ = lp.shape
    U = U1 - 1
    blank_lp = lp[..., blank]                                               # [B, T, U+1]
    if U > 0:
        emit_lp = lp[:, :, :U, :].gather(-1, labels[:, None, :U, None].expand(B, T, U, 1)).squeeze(-1)
    else:
        emit_lp = lp.new_zeros(B, T, 0)
    zero = lp.new_zeros(B, T, 1)
    C = torch.cat([zero, emit_lp.cumsum(-1)], dim=-1)                       # [B, T, U+1]

    alphas = [C[:, 0]]                                                      # t = 0: only emits
    for t in range(1, T):
        a = alphas[-1] + blank_lp[:, t - 1]
        alphas.append(C[:, t] + torch.logcumsumexp(a - C[:, t], dim=-1))
    alpha = torch.stack(alphas, dim=1)                                      # [B, T, U+1]
    b = torch.arange(B, device=lp.device)
    tl, ul = act_lens.long() - 1, label_lens.long()
    return -(alpha[b, tl, ul] + blank_lp[b, tl, ul])


class FastRNNTLoss(nn.Module):
    """Drop-in for NeMo's inner `_loss`: forward(acts, labels, act_lens, label_lens) -> [B]."""

    def __init__(self, blank: int):
        super().__init__()
        self.blank = blank
        self.reduction = None            # NeMo's RNNTLoss sets/resets this around the call

    def forward(self, acts, labels, act_lens, label_lens):
        return rnnt_nll(acts, labels, act_lens, label_lens, self.blank)


if __name__ == "__main__":
    import time
    from nemo.collections.asr.losses.rnnt_pytorch import RNNTLossPytorch

    torch.manual_seed(0)
    B, T, U, V = 3, 13, 6, 9
    acts = torch.randn(B, T, U + 1, V, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, V - 1, (B, U))
    al, ll = torch.tensor([13, 9, 11]), torch.tensor([6, 3, 0])
    ref = RNNTLossPytorch(blank=V - 1, reduction="none")(acts=acts, labels=labels, act_lens=al, label_lens=ll)
    mine = rnnt_nll(acts.double(), labels, al, ll, V - 1)
    g1 = torch.autograd.grad(ref.sum(), acts)[0]
    g2 = torch.autograd.grad(mine.sum(), acts)[0]
    print("loss ref ", ref.detach().numpy().round(6))
    print("loss mine", mine.detach().numpy().round(6))
    print("max |dloss|", float((ref - mine).abs().max()), " max |dgrad|", float((g1 - g2).abs().max()))
    assert torch.allclose(ref, mine.to(ref.dtype), atol=1e-6) and torch.allclose(g1, g2, atol=1e-6)

    dev = "cuda"
    B, T, U, V = 16, 250, 100, 2049
    acts = torch.randn(B, T, U + 1, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V - 1, (B, U), device=dev)
    al = torch.full((B,), T, device=dev); ll = torch.full((B,), U, device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss = rnnt_nll(acts, labels, al, ll, V - 1).mean(); loss.backward()
    torch.cuda.synchronize()
    print(f"realistic batch 16 x 20 s, U=100, V=2049: fwd+bwd {time.perf_counter() - t0:.2f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("SELFTEST OK")
