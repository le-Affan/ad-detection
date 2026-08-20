"""3D ResNet-18 trunk with the SDD's two heads.

Structure follows the Toqa277 reference: a textbook ResNet-18 rebuilt with 3D
convolutions, ending in AdaptiveAvgPool3d((1,1,1)) -> a 512-d vector. Because
the pool is adaptive there are no hardcoded spatial dimensions anywhere in the
model, so input size is free -- 96^3 today, something else later, no code
change. That is deliberate and worth preserving; the NYUMedML reference hard-
codes a 5*5*5 flatten and only works at one input size.

Trunk and heads are separate modules, per NYUMedML's composition idiom. The
trunk is the reusable part: the same 512-d vector feeds DEMO and ALZ here, and
is the natural place to tap features for the downstream fusion model. Grad-CAM
should target `trunk.layer4`.

Heads, per the SDD:

    DEMO : Linear(512, 1), continuous severity 0-2, trained with MSE
    ALZ  : Linear(512, 1), AD probability, trained with BCE

ON THE SIGMOID. forward() returns the ALZ *logit*, not the probability, and
`alz_prob` is provided separately. This is not a deviation from the SDD -- the
sigmoid is still the model's output nonlinearity -- but BCE must be computed
from the logit via BCEWithLogitsLoss, which folds the sigmoid into the loss
using the log-sum-exp trick. Applying sigmoid first and then plain BCELoss is
numerically unstable and will produce inf/NaN gradients once the model becomes
confident. Both are exposed so callers cannot get this wrong by accident.
"""
import torch
import torch.nn as nn


def conv3x3x3(cin, cout, stride=1):
    return nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Standard ResNet basic block, 3D. Two 3x3x3 convs plus identity."""

    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(cout, cout)
        self.bn2 = nn.BatchNorm3d(cout)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(out + idt)


class ResNet3DTrunk(nn.Module):
    """ResNet-18 in 3D, returning a 512-d embedding.

    Input is (B, 1, D, H, W) for any D/H/W large enough to survive the stride-32
    downsampling; output is (B, 512).
    """

    def __init__(self, in_channels=1, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.cin = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(widths[0], blocks[0], 1)
        self.layer2 = self._make_layer(widths[1], blocks[1], 2)
        self.layer3 = self._make_layer(widths[2], blocks[2], 2)
        self.layer4 = self._make_layer(widths[3], blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_features = widths[3]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, cout, n, stride):
        downsample = None
        if stride != 1 or self.cin != cout:
            downsample = nn.Sequential(
                nn.Conv3d(self.cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(cout))
        layers = [BasicBlock(self.cin, cout, stride, downsample)]
        self.cin = cout
        layers += [BasicBlock(cout, cout) for _ in range(1, n)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.pool(x), 1)


class DualHeadNet(nn.Module):
    """Trunk plus the DEMO regression head and the ALZ binary head."""

    def __init__(self, in_channels=1, dropout=0.0):
        super().__init__()
        self.trunk = ResNet3DTrunk(in_channels)
        d = self.trunk.out_features
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.demo_head = nn.Linear(d, 1)
        self.alz_head = nn.Linear(d, 1)

    def forward(self, x, return_features=False):
        f = self.drop(self.trunk(x))
        out = {
            "demo": self.demo_head(f).squeeze(-1),       # continuous severity
            "alz_logit": self.alz_head(f).squeeze(-1),   # feed to BCEWithLogitsLoss
        }
        if return_features:
            out["features"] = f
        return out

    @staticmethod
    def alz_prob(out):
        """AD probability in [0, 1] from a forward() result."""
        return torch.sigmoid(out["alz_logit"])


def dual_head_loss(out, batch, lam_demo=1.0, lam_alz=1.0):
    """Combined objective: lam1 * MSE(DEMO) + lam2 * BCE(ALZ).

    BCEWithLogitsLoss consumes the raw logit; see the module docstring.
    """
    mse = nn.functional.mse_loss(out["demo"], batch["demo"])
    bce = nn.functional.binary_cross_entropy_with_logits(out["alz_logit"], batch["alz"])
    total = lam_demo * mse + lam_alz * bce
    # detach for reporting: these are logged, never backpropagated through
    return total, {"mse": float(mse.detach()), "bce": float(bce.detach())}
