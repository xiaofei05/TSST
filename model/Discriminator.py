import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Directly borrowed from
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        filter_sizes = [2, 4, 6, 8]
        num_filters = [config.num_filters] * len(filter_sizes)

        self.emb_size = config.embedding_size
        self.feature_size = sum(num_filters)

        self.convs = nn.ModuleList([
            SpectralNorm(nn.Conv2d(1, n, (f, self.emb_size))) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.feature2out = nn.Sequential(
            SpectralNorm(nn.Linear(self.feature_size, 64)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(64, 2+1))) # M+1 classes

        self.dropout = nn.Dropout(config.dropout)
        self.activ = nn.LeakyReLU(0.2)

    def forward(self, inps):
        # inps: (B, L, emb_size)
        feature = self.get_feature(inps)
        logits = self.feature2out(self.dropout(feature))
        return logits
        # probs = F.softmax(logits, dim=-1)
        # return logits, probs

    def get_feature(self, inps):
        embs = inps.unsqueeze(1) # (B, 1, L, emb_size)

        # features: (B, len(filter_sizes), length)
        #   each feature (B, filter_num(64), length)
        features = [self.activ(conv(embs)).squeeze(3) for conv in self.convs]
        # pools: (B, filter_size)
        pools = [F.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in features]
        h = torch.cat(pools, 1)  # (B, feature_size)
        return h