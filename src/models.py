"""Model definitions (Q1-Q3) plus parameter and MAC counting."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden=(256, 256), bn=True, in_dim=3 * 32 * 32, n_classes=10):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNA(nn.Module):
    """3x3 convs 64-64-M-128-128-M-192-192-M, BN+ReLU, global average pool, FC-10."""
    CFG = [64, 64, "M", 128, 128, "M", 192, 192, "M"]

    def __init__(self, n_classes=10):
        super().__init__()
        layers, c = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c, n_classes)

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.fc(self.features(x))


def _cbr(cin, cout, stride):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def _tbr(cin, cout):
    return [nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class ConvAE(nn.Module):
    """Fully convolutional AE. Only the bottleneck channel count c changes between models:
    z has shape (c,4,4), i.e. d = 16c dimensions."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.encoder = nn.Sequential(*_cbr(3, 64, 1), *_cbr(64, 128, 2), *_cbr(128, 256, 2),
                                     *_cbr(256, 256, 2), nn.Conv2d(256, c, 1))
        self.decoder = nn.Sequential(nn.Conv2d(c, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     *_tbr(256, 256), *_tbr(256, 128), *_tbr(128, 64),
                                     nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


MODEL_SPECS = {
    "mlp_s": dict(kind="mlp", hidden=(128, 128), bn=True),
    "mlp_b": dict(kind="mlp", hidden=(256, 256), bn=True),
    "mlp_l": dict(kind="mlp", hidden=(512, 512), bn=True),
    "mlp_xl": dict(kind="mlp", hidden=(1024, 1024), bn=True),
    "mlp_b_plain": dict(kind="mlp", hidden=(256, 256), bn=False),
    "mlp_mem": dict(kind="mlp", hidden=(2048, 2048), bn=False),
    "cnn_a": dict(kind="cnn"),
}


def build_model(name):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "mlp":
        return MLP(hidden=spec["hidden"], bn=spec["bn"])
    return CNNA()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_macs(model, input_shape=(1, 3, 32, 32)):
    """Multiply-accumulates per image for Linear / Conv2d / ConvTranspose2d layers."""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs[0] += k * out[0].numel()
        elif isinstance(m, nn.ConvTranspose2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.out_channels // m.groups)
            macs[0] += k * inp[0][0].numel()

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(input_shape, device=dev))
    model.train(was_training)
    for h in hs:
        h.remove()
    return macs[0]


if __name__ == "__main__":
    for n in MODEL_SPECS:
        m = build_model(n)
        print(f"{n:12s} params={count_params(m):>10,d}  MACs/img={count_macs(m):>12,d}")
    for c in (2, 16, 128):
        m = ConvAE(c)
        print(f"ae_c{c:<8d} d={16 * c:<5d} params={count_params(m):>10,d}")
