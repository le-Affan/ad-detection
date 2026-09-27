"""Path A temporal model: the DualHeadNet trunk per visit, a temporal module
over the visit sequence, and the same two heads.

    volumes (B,4,1,D,H,W), mask (B,4), days (B,4)
      -> ResNet3DTrunk on USABLE visits only       -> (B,4,d), zeros where masked
      -> temporal module (mask-aware)              -> (B,h) subject embedding
      -> demo_head / alz_head                      -> {"demo", "alz_logit"}

The output dict has the same keys as DualHeadNet's, so dual_head_loss,
m3_train's metrics and the downstream fusion all read it unchanged. model.py
is not modified; the trunk class and its ARCHS presets are imported.

HOW MASKING IS ENFORCED -- three independent layers, none relying on zeros:

  1. The trunk never sees a masked slot. Usable volumes are gathered into a
     flat (N,1,D,H,W) batch, encoded, and scattered back; masked positions are
     zero embeddings that no volume produced. This also means a subject with
     two visits costs two trunk passes, not four. BatchNorm statistics are
     therefore computed over real scans only.
  2. Each temporal module consumes the mask directly (below), so a masked
     embedding cannot reach the output even if it were non-zero.
  3. The loss is per subject, so there is no per-slot term to mask.

`forward_embeddings` exposes stage 2 alone, which is what C1 trains on (cached
trunk embeddings) and what the invariance test checks: random values written
into masked slots must leave the output bit-identical.

TIME. `days` is measured from each subject's first usable visit and is fed as
years. Real visit spacing drifts from the nominal slots (an "m24" scan landed
at 36 months), so the model sees the actual interval, not the slot label.

TEMPORAL MODULES
  delta : [e_last, e_last - e_first, t_last - t_first] -> MLP. The indices are
          the first and last USABLE visits, found from the mask. One visit
          gives a zero delta and zero interval -- the static case.
  gru   : usable visits compacted to the front of the sequence in time order
          (so a 1011 pattern becomes three consecutive steps), each step fed
          [proj(e), t, gap since previous visit], run as a packed sequence of
          per-subject length; the final hidden state is the output.
  attn  : proj(e) + a learned embedding of t, a small transformer encoder with
          src_key_padding_mask = ~mask, then mean-pooling over usable visits.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from model import ARCHS, ResNet3DTrunk

DAYS_PER_YEAR = 365.25


def _check_mask(mask):
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    if not mask.any(dim=1).all():
        raise ValueError("every subject needs at least one usable visit")


class DeltaTemporal(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d + 1, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout))
        self.out_features = hidden

    def forward(self, emb, mask, t):
        idx = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        big = mask.shape[1]
        first = torch.where(mask, idx, big).min(dim=1).values
        last = torch.where(mask, idx, -1).max(dim=1).values
        b = torch.arange(emb.shape[0], device=emb.device)
        e_first, e_last = emb[b, first], emb[b, last]
        dt = (t[b, last] - t[b, first]).unsqueeze(1)
        return self.net(torch.cat([e_last, e_last - e_first, dt], dim=1))


class GRUTemporal(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden + 2, hidden, batch_first=True)
        self.out_features = hidden

    def forward(self, emb, mask, t):
        # compact usable visits to the front, keeping slot (= time) order
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        e = torch.gather(emb, 1, order.unsqueeze(-1).expand_as(emb))
        tt = torch.gather(t, 1, order)
        lengths = mask.sum(dim=1)
        gap = torch.diff(tt, dim=1, prepend=tt[:, :1])
        x = torch.cat([self.proj(e), tt.unsqueeze(-1), gap.unsqueeze(-1)], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]


class AttnTemporal(nn.Module):
    def __init__(self, d, hidden, dropout, layers=1, heads=4):
        super().__init__()
        self.proj = nn.Linear(d, hidden)
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden,
                                           dropout=dropout, batch_first=True)
        # enable_nested_tensor=False: the nested fast path drops padded tokens
        # itself, which is fine, but it is inference-only and would make train
        # and eval take different code paths through the mask.
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_features = hidden

    def forward(self, emb, mask, t):
        x = self.proj(emb) + self.time(t.unsqueeze(-1))
        x = self.enc(x, src_key_padding_mask=~mask)
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(dim=1) / w.sum(dim=1)


TEMPORAL = {"delta": DeltaTemporal, "gru": GRUTemporal, "attn": AttnTemporal}


class TemporalDualHeadNet(nn.Module):
    """Shared per-visit trunk + temporal module + DEMO/ALZ heads."""

    def __init__(self, temporal="gru", arch="resnet18w", hidden=128, dropout=0.3,
                 in_channels=1):
        super().__init__()
        if temporal not in TEMPORAL:
            raise ValueError(f"unknown temporal {temporal!r}; choose from {sorted(TEMPORAL)}")
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
        self.temporal_kind, self.arch = temporal, arch
        self.trunk = ResNet3DTrunk(in_channels, **ARCHS[arch])
        d = self.trunk.out_features
        self.temporal = TEMPORAL[temporal](d, hidden, dropout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.demo_head = nn.Linear(self.temporal.out_features, 1)
        self.alz_head = nn.Linear(self.temporal.out_features, 1)

    def encode(self, volumes, mask):
        """(B,4,1,D,H,W) -> (B,4,d). Only usable slots pass through the trunk."""
        _check_mask(mask)
        feats = self.trunk(volumes[mask])                        # (N, d)
        emb = feats.new_zeros(mask.shape + (feats.shape[-1],))
        emb[mask] = feats
        return emb

    def forward_embeddings(self, emb, mask, days, return_features=False):
        _check_mask(mask)
        t = days.to(emb.dtype) / DAYS_PER_YEAR
        f = self.drop(self.temporal(emb, mask, t))
        out = {"demo": self.demo_head(f).squeeze(-1),
               "alz_logit": self.alz_head(f).squeeze(-1)}
        if return_features:
            out["features"] = f
        return out

    def forward(self, volumes, mask, days, return_features=False):
        return self.forward_embeddings(self.encode(volumes, mask), mask, days, return_features)

    @staticmethod
    def alz_prob(out):
        return torch.sigmoid(out["alz_logit"])

    def load_trunk(self, ckpt_path, map_location="cpu"):
        """Initialise the trunk from a DualHeadNet checkpoint (m3_train's best.pt).

        Only `trunk.*` is taken -- the static model's heads map one scan to a
        target and have no meaning after a temporal module. The architecture
        must match exactly; a width mismatch raises rather than half-loading.
        """
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ck.get("model", ck)
        trunk = {k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")}
        if not trunk:
            raise ValueError(f"{ckpt_path} has no trunk.* weights")
        self.trunk.load_state_dict(trunk, strict=True)
        return ck.get("args", {})
