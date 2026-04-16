
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PosEncoding(nn.Module):
    def __init__(self, num_freq=6):
        super().__init__()
        self.num_freq = num_freq

    def forward(self, coords):
        """
        coords: [N, 2]  (x, y in [-1,1])
        """
        out = [coords]

        for i in range(self.num_freq):
            freq = 2.0 ** i * math.pi
            out.append(torch.sin(coords * freq))
            out.append(torch.cos(coords * freq))

        return torch.cat(out, dim=-1)


class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MetaUpsample(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim=64,
            num_freq=6,
            use_guidance=False
    ):
        super().__init__()

        self.use_guidance = use_guidance

        self.encoder = PosEncoding(num_freq=num_freq)

        # 输入维度 = (2 + 4*num_freq)
        mlp_in_dim = 2 + 4 * num_freq

        if use_guidance:
            # 如果有 guide，输入增加通道信息
            mlp_in_dim += in_channels

        self.mlp = MetaMLP(mlp_in_dim, hidden_dim, out_dim=in_channels)

        self.conv_out = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, feat_lr, scale, guide_hr=None):
        """
        feat_lr: [B,C,Hl,Wl]
        guide_hr: [B,C,Hh,Wh] (optional)
        """

        B, C, H, W = feat_lr.shape
        device = feat_lr.device

        H_out = int(round(H * scale))
        W_out = int(round(W * scale))

        feat_up = F.interpolate(
            feat_lr,
            size=(H_out, W_out),
            mode='bilinear',
            align_corners=False
        )

        ys = torch.linspace(-1, 1, H_out, device=device)
        xs = torch.linspace(-1, 1, W_out, device=device)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        coords = coords.view(-1, 2)  # [N,2]

        encoded = self.encoder(coords)  # [N, D]

        if self.use_guidance and guide_hr is not None:
            guide = F.interpolate(
                guide_hr,
                size=(H_out, W_out),
                mode='bilinear',
                align_corners=False
            )

            guide = guide.permute(0, 2, 3, 1).reshape(-1, C)

            encoded = torch.cat([encoded, guide], dim=-1)

        weights = self.mlp(encoded)  # [N,C]

        weights = weights.view(1, H_out, W_out, C)
        weights = weights.permute(0, 3, 1, 2)

        # =========================

        out = feat_up * weights + feat_up

        out = self.conv_out(out)

        return out