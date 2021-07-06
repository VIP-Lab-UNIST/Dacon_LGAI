import time
import datetime
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

# --- Build dense --- #
class DeformModule(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(DeformModule, self).__init__()
        self.conv = DeformConv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x,out), dim=1)
        return out
        
class VanilaModule(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(VanilaModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x,out), dim=1)
        return out

# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, flag=False):
        super(RDB, self).__init__()
        _in_channels = in_channels
        growth_rate = in_channels // 4
        modules = []
        if flag == True :
            for i in range(3):
                modules.append(VanilaModule(_in_channels, growth_rate))
                _in_channels += growth_rate
            modules.append(DeformModule(_in_channels, growth_rate))
            _in_channels += growth_rate
        else :
            for i in range(4):
                modules.append(VanilaModule(_in_channels, growth_rate))
                _in_channels += growth_rate

        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class Net(nn.Module):
    def __init__(self, in_channel=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, padding=5, padding_mode='reflect')
        self.DeformRDB1 = RDB(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1)
        self.DeformRDB2 = RDB(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 4, padding=1)
        self.DeformRDB3 = RDB(128, flag=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride = 4, padding=1)
        self.DeformRDB4 = RDB(256, flag=True)
        self.bottle = RDB(256, flag=True)

        self.deconv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.DeformRDB5 = RDB(128, flag=True)
        self.deconv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.DeformRDB6 = RDB(64)
        self.deconv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.DeformRDB7 = RDB(32)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        feat0 = self.conv1(x) # 32 ch
        feat  = self.DeformRDB1(feat0) # 32 ch
        feat1 = self.conv2(feat)       # 64 ch, 1/2 size
        feat  = self.DeformRDB2(feat1) # 64 ch, 1/2 size

        feat2 = self.conv3(feat)       # 128 ch, 1/8 size
        feat  = self.DeformRDB3(feat2) # 128 ch, 1/8 size
        
        feat  = self.conv4(feat)      # 256 ch, 1/32 size
        feat  = self.DeformRDB4(feat) # 256 ch, 1/32 size
        feat  = self.bottle(feat)

        feat  = F.interpolate(feat, size=(x.size(2)//8, x.size(3)//8), mode='bilinear') # 128ch, 1/8 size
        feat  = feat2 + self.deconv4(feat) # 64ch, 1/8 size
        feat  = self.DeformRDB5(feat) # 64ch, 1/8 size

        feat  = F.interpolate(feat, size=(x.size(2)//2, x.size(3)//2), mode='bilinear') # 128ch, 1/2 size
        feat  = feat1 + self.deconv3(feat) # 64ch, 1/2 size
        feat  = self.DeformRDB6(feat) # 64ch, 1/2 size

        feat  = F.interpolate(feat, size=(x.size(2), x.size(3)), mode='bilinear') # 64ch, orig size
        feat  = feat0 + self.deconv2(feat) # 32ch, orig size
        feat  = self.DeformRDB7(feat) # 64ch, 1/2 size
        out   = self.conv_out(feat)
        return out

# --- Main model  --- #
class SatPyramid(nn.Module):
    def __init__(self, N=25, theshold=0.7, kernel_size=15):
        super(SatPyramid, self).__init__()
        self.N = N
        self.kernel_size = kernel_size
        self.Thres = nn.Threshold(theshold, 0)
        self.avgpool = nn.AvgPool2d(kernel_size, stride =1, padding=(kernel_size-1)//2)

    def forward(self, x):
        feat = x
        out = feat * 0
        for iters in range(self.N):
            feat = self.Thres(feat)
            feat = self.avgpool(feat)
            out += feat
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding=1)

        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1x1 = nn.Conv2d(256, 1, kernel_size=1, stride = 1, padding=0)

    def forward(self, x):
        feat1 = F.leaky_relu(self.conv1(x))
        feat2 = F.leaky_relu(self.bn2(self.conv2(feat1)))
        feat3 = F.leaky_relu(self.bn3(self.conv3(feat2)))
        feat4 = F.leaky_relu(self.bn4(self.conv4(feat3)))
        prob = self.conv1x1(feat4)
        return prob

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

