import torch


class FrozenEncoder:
    def __init__(self, checkpoint, device='cuda'):
        from cellpose import models
        cp = models.CellposeModel(
            gpu=device.startswith('cuda'), pretrained_model=str(checkpoint), device=torch.device(device))
        cp.net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.net = cp.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feat = {}
        self.net.encoder.neck.register_forward_hook(lambda m, i, o: self._feat.__setitem__('f', o.detach()))

    @torch.no_grad()
    def tokens(self, x):
        x = x[:, None]
        x = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1).to(self.device).to(torch.bfloat16)
        self.net(x)
        return self._feat['f'].float()
