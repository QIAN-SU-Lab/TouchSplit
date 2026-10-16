import torch
import torch.nn as nn
import torch.nn.functional as F

TILE = 256


class TouchSplitHead(nn.Module):
    def __init__(self, c=256, h=128, size=TILE):
        super().__init__()
        self.size = size
        self.b0 = nn.Sequential(nn.Conv2d(c, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, h, 3, padding=1), nn.GELU())
        chs = [h, h // 2, h // 4, h // 8]
        self.ups = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chs[i], chs[i + 1] * 4, 1), nn.PixelShuffle(2),
            nn.Conv2d(chs[i + 1], chs[i + 1], 3, padding=1), nn.GELU()) for i in range(3)])
        self.sides = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in chs])
        self.fuse = nn.Conv2d(4, 1, 1)
        self.dhead = nn.Conv2d(chs[-1], 1, 3, padding=1)

    def forward(self, x):
        f = self.b0(x)
        outs = [self.sides[0](f)]
        for i, up in enumerate(self.ups):
            f = up(f)
            outs.append(self.sides[i + 1](f))
        outs = [o if o.shape[-1] == self.size else
                F.interpolate(o, (self.size, self.size), mode='bilinear', align_corners=False) for o in outs]
        return [o[:, 0] for o in outs], self.fuse(torch.cat(outs, 1))[:, 0], torch.sigmoid(self.dhead(f))[:, 0]

