import torch
import torch.nn as nn
import torch.nn.functional as F
class ATACBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super(ATACBasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class S2DBottleneckBlock(nn.Module):
    def __init__(self, channels, planes, num_groups=32, dilation=1):
        super(S2DBottleneckBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes*2)

        self.conv2 = nn.Conv1d(planes*2, planes*2, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation,
                               bias=False, groups=num_groups)
        self.bn2 = nn.BatchNorm1d(planes*2)

        self.conv3 = nn.Conv1d(planes*2, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class S2DModule(nn.Module):
    def __init__(self, base_channels=96, num_groups=24):
        super(S2DModule, self).__init__()

        self.shared_proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ATACBasicBlock(64, base_channels, stride=1, num_groups=16),
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.local_branch = nn.Sequential(
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=1),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=1),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=1),
            nn.Dropout(0.05),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=2),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=2),
            ATACBasicBlock(base_channels, base_channels, stride=1, num_groups=num_groups),
        )

        self.medium_branch = nn.Sequential(
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=2),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=3),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=5),
            nn.Dropout(0.05),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=7),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=9),
            ATACBasicBlock(base_channels, base_channels, stride=1, num_groups=num_groups),
        )

        self.global_branch = nn.Sequential(
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=5),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=7),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=11),
            nn.Dropout(0.05),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=13),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=17),
            ATACBasicBlock(base_channels, base_channels, stride=1, num_groups=num_groups),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(base_channels * 3, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            ATACBasicBlock(base_channels * 2, base_channels * 2, stride=1, num_groups=num_groups*2),
            nn.Dropout(0.05),
            ATACBasicBlock(base_channels * 2, base_channels * 2, stride=1, num_groups=num_groups*2),
            nn.Dropout(0.05),

            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),

            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=1),
            S2DBottleneckBlock(base_channels, base_channels//4, num_groups=num_groups, dilation=1),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.output_proj = nn.Sequential(
            nn.Conv1d(base_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=False)
        )

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, sparse_atac):
        identity = sparse_atac
        target_length = sparse_atac.shape[2]

        x = self.shared_proj(sparse_atac)

        x_down = self.downsample(x)

        local_feat = self.local_branch(x_down)
        medium_feat = self.medium_branch(x_down)
        global_feat = self.global_branch(x_down)

        fused = torch.cat([local_feat, medium_feat, global_feat], dim=1)
        fused = self.fusion(fused)

        fused_up = self.upsample(fused)

        if fused_up.shape[2] != target_length:
            fused_up = F.interpolate(fused_up, size=target_length, mode='linear', align_corners=False)

        enhancement = self.output_proj(fused_up)

        dense_atac = identity + self.residual_weight * enhancement

        return dense_atac
