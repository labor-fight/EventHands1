"""EventHPE official modules (event_pose_estimation/model.py at 2af451f), vendored: `TemporalEncoder` and `Regressor`
verbatim; `resnet50_encoder` is `EventTrackNet.resnet50_encoder` as a module-level function (its body verbatim).
`EventTrackNet.forward` (SMPL, 24 joints) is restated for MANO in semkine/cmp_ehpe.py. See README.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as func  # noqa: F401
from torchvision.models import resnet50


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
            batch_first=True
        )

        self.h0_size = n_layers * (bidirectional+1)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x, hidden_feats):
        # x: [B, T, F]
        # hidden_feats: [B, 2048]
        h0 = hidden_feats.unsqueeze(0).repeat(self.h0_size, 1, 1)
        B, T, F = x.shape
        y, _ = self.gru(x, h0)  # y: [B, T, F]
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(B, T, F)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        return y


class Regressor(nn.Module):
    def __init__(self, pose_dim=24*6):
        super(Regressor, self).__init__()
        # self.fc1 = nn.Linear(2048, 1024)
        # self.decpose = nn.Linear(1024, pose_dim)
        # self.dectrans = nn.Linear(1024, 3)
        self.decpose = nn.Linear(2048, pose_dim)
        self.dectrans = nn.Linear(2048, 3)


    def forward(self, x):
        # x: [B, T, 2048]
        B, T, F = x.size()
        x = x.view(-1, F)
        # x = func.dropout(self.fc1(x))
        x1 = self.decpose(x)  # [B*T, pose_dim]
        x2 = self.dectrans(x)  # [B*T, 3]
        delta_pose = x1.view(B, T, x1.size(-1))
        delta_tran = x2.view(B, T, x2.size(-1))
        return delta_pose, delta_tran


def resnet50_encoder(input_channel):
    model = resnet50(pretrained=False)
    model.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential()
    return model
