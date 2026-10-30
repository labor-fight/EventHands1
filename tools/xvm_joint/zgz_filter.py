"""Original S39 adaptive postfilter around one XJ raw-packet proposal.

The core consumes the new event prefix once.  Its full-window original LNES
provides the historical empty-image gate, while its latest 50 ms raw event
count selects the unmodified ramp1000 gains.  The evaluator owns feedback of
the filtered output; no target or packet beta is inspected by this wrapper.
"""
from __future__ import annotations

import numbers

import torch
from torch import nn

from semkine.anchored import AdaptiveFilter


class _ProposalProxy(nn.Module):
    """Give AdaptiveFilter the already-computed proposal without another call."""

    def __init__(self):
        super().__init__()
        self.proposal = None

    def forward(self, x, prevpos, betas=None, camera_K=None):
        if self.proposal is None:
            raise RuntimeError('No current XJ proposal is installed')
        return self.proposal


class S39PacketFilter(nn.Module):
    """Raw-event evaluator interface with the original AdaptiveFilter math.

    ``core`` must implement forward_packet/reset_state, and expose the last
    full-window BHWC LNES plus the latest-50ms integer count.  A state snapshot
    is optional unless snapshot_state/restore_state is used for engineering
    timing.  No get_state is exposed: evalx would otherwise retain the entire
    core event field once per evaluated frame.
    """

    encoder_name = 'xvm_joint_s39_packet_filter'
    xyz_assoc_prev = 'self'

    def __init__(self, core, spec):
        super().__init__()
        self.core = core
        self._proposal_proxy = _ProposalProxy()
        self.filter = AdaptiveFilter.from_spec(self._proposal_proxy, spec)

    def train(self, mode=True):
        super().train(False)
        return self

    def set_hand_context(self, betas, camera_K):
        # The frozen XJ core obtains K from every packet and fixes beta0.  If
        # a context hook exists, never forward the evaluator's score-only beta.
        if hasattr(self.core, 'set_hand_context'):
            self.core.set_hand_context(None, camera_K)

    def reset_state(self, prev):
        self._proposal_proxy.proposal = None
        self.core.reset_state(prev)
        self.filter.reset_state(prev)

    def snapshot_state(self):
        if self._proposal_proxy.proposal is not None:
            raise RuntimeError('Cannot snapshot a partially filtered packet')
        return dict(core=self.core.snapshot_state(), filter=self.filter.get_state())

    def restore_state(self, state):
        if self._proposal_proxy.proposal is not None:
            raise RuntimeError('Cannot restore a partially filtered packet')
        self.core.restore_state(state['core'])
        self.filter.set_state(state['filter'])

    def spec(self):
        return self.filter.spec()

    @property
    def last_lnes(self):
        return self.core.last_lnes

    @property
    def last_current_count(self):
        return self.core.last_current_count

    @property
    def last_audit(self):
        return self.core.last_audit

    def forward(self, batch):
        return self.forward_packet(batch)

    @torch.no_grad()
    def forward_packet(self, batch):
        previous = batch.prev_state
        proposal = self.core.forward_packet(batch)
        x, count = self.core.last_lnes, self.core.last_current_count
        if (not torch.is_tensor(x) or x.ndim != 4 or x.shape[0] != previous.shape[0]
                or x.shape[-1] != 2 or x.device != previous.device):
            raise ValueError('Core must expose the current full-window BHWC LNES')
        if not isinstance(count, numbers.Integral) or count < 0:
            raise ValueError('Core must expose the latest50ms nonnegative integer event count')
        if proposal.shape != previous.shape or proposal.device != previous.device:
            raise ValueError('Core proposal and evaluator previous-state contracts differ')
        self._proposal_proxy.proposal = proposal
        try:
            # ProxyTrk only returns proposal. AdaptiveFilter applies its own
            # SO(3)/linear update, count gains, velocity and exact empty hold.
            return self.filter(x, previous, n_events=int(count))
        finally:
            self._proposal_proxy.proposal = None
