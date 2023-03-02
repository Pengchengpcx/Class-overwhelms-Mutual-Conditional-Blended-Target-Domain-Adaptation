import torch
import torch.nn as nn

class StyleInjectTtoS(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, concate_x):
        N, C, H, W = concate_x.size()
        bs = int(N/2)
        if self.training:
            x_s, x_t = concate_x.chunk(2, dim=0)
            # get source content in target style
            x_s = x_s.view(bs, C, -1)
            mean_s = x_s.mean(-1, keepdim=True)
            var_s = x_s.var(-1, keepdim=True)

            x_t = x_t.view(bs, C, -1).detach()
            mean_t = x_t.mean(-1, keepdim=True)
            var_t = x_t.var(-1, keepdim=True)

            x_s = (x_s - mean_s) / (var_s + self.eps).sqrt()
            idx_swap = torch.randperm(bs)
            alpha = torch.rand(bs, 1, 1).cuda()

            # remix styles
            mean = alpha * mean_s + (1 - alpha) * mean_t[idx_swap]
            var = alpha * var_s + (1 - alpha) * var_t[idx_swap]

            x_st = x_s * (var + self.eps).sqrt() + mean
            x_st = x_st.view(bs, C, H, W)
            x_t = x_t.view(bs, C, H, W)

            concate_x = torch.cat((x_st, x_t), dim=0)

            return concate_x

        return concate_x