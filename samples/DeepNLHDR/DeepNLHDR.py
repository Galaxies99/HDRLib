import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, ks, stride):
        super().__init__()
        # Custom Padding Calculation
        if isinstance(ks, tuple):
            k_h, k_w = ks
        else:
            k_h = ks
            k_w = ks
        if isinstance(stride, tuple):
            s_h, s_w = stride
        else:
            s_h = stride
            s_w = stride
        pad_h, pad_w = k_h - s_h, k_w - s_w
        pad_up, pad_left = pad_h // 2, pad_w // 2
        pad_down, pad_right= pad_h - pad_up, pad_w - pad_left
        self.pad = nn.ReflectionPad2d([pad_left, pad_right, pad_up, pad_down])
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=ks, stride=stride, bias=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, ks=3, stride=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(dim, momentum=0.9),
            PaddedConv2d(dim, dim, ks=ks, stride=stride),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(dim, momentum=0.9),
            PaddedConv2d(dim, dim, ks=ks, stride=stride),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return x + y


class ResidualBlocks(nn.Module):
    def __init__(self, block_num, dim, ks=3, stride=1):
        super().__init__()
        self.block_num = block_num
        self.res_blocks = nn.Sequential()
        for i in range(self.block_num):
            self.res_blocks.add_module('Residual' + str(i), ResidualBlock(dim, ks=ks, stride=stride))
            

    def forward(self, x):
        y = x
        for i in range(self.block_num):
            y = self.res_blocks[i](y)
        return y

class NonLocal(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.theta = PaddedConv2d(channel, channel // 2, 1, 1)
        self.phi = PaddedConv2d(channel, channel // 2, 1, 1)
        self.g = PaddedConv2d(channel, channel // 2, 1, 1)
        self.final = PaddedConv2d(channel // 2, channel, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).contiguous().view(b, c // 2, h * w).permute(0, 2, 1).contiguous()
        phi = self.phi(x).contiguous().view(b, c // 2, h * w)
        g = self.g(x).contiguous().view(b, c // 2, h * w).permute(0, 2, 1).contiguous()
        theta_phi = torch.matmul(theta, phi)
        softmax_res = F.softmax(theta_phi, dim=-1)
        theta_phi_g = torch.matmul(softmax_res, g)
        theta_phi_g = theta_phi_g.permute(0, 2, 1).contiguous().view(b, c // 2, h, w)
        theta_phi_g = self.final(theta_phi_g)
        return x + theta_phi_g


class GlobalNonLocal(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.non_local = NonLocal(channel)

    def forward(self, x):
        _, _, h, w = x.shape
        y = self.pool(x)
        y = self.non_local(y)
        y = F.interpolate(y, size=[h, w])
        return y



class DeepNLHDR(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.image_h, self.image_w = configs.image_size
        self.c_dim = configs.c_dim
        self.num_res_blocks = 9
        self.gf_dim = 64
        self.num_shots = configs.num_shots

        self.encoder1_1 = nn.Sequential(
            PaddedConv2d(self.c_dim * 2, self.gf_dim, ks=5, stride=2),
            nn.ReLU(True)
        )
        self.encoder1_2 = nn.Sequential(
            PaddedConv2d(self.gf_dim, self.gf_dim * 2, ks=5, stride=2),
            nn.BatchNorm2d(self.gf_dim * 2, momentum=0.9),
            nn.ReLU(True)
        )

        self.encoder2_1 = nn.Sequential(
            PaddedConv2d(self.c_dim * 2, self.gf_dim, ks=5, stride=2),
            nn.ReLU(True)
        )
        self.encoder2_2 = nn.Sequential(
            PaddedConv2d(self.gf_dim, self.gf_dim * 2, ks=5, stride=2),
            nn.BatchNorm2d(self.gf_dim * 2, momentum=0.9),
            nn.ReLU(True)
        )

        self.encoder3_1 = nn.Sequential(
            PaddedConv2d(self.c_dim * 2, self.gf_dim, ks=5, stride=2),
            nn.ReLU(True)
        )
        self.encoder3_2 = nn.Sequential(
            PaddedConv2d(self.gf_dim, self.gf_dim * 2, ks=5, stride=2),
            nn.BatchNorm2d(self.gf_dim * 2, momentum=0.9),
            nn.ReLU(True)
        )

        self.merger = nn.Sequential(
            PaddedConv2d(self.gf_dim * 2 * 3, self.gf_dim * 4, ks=5, stride=2),
            nn.BatchNorm2d(self.gf_dim * 4, momentum=0.9),
            nn.ReLU(True)
        )

        self.residual_blocks = ResidualBlocks(block_num=self.num_res_blocks, dim=self.gf_dim * 4, ks=3, stride=1)
        self.non_local = GlobalNonLocal(self.gf_dim * 4)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.gf_dim * 4 * 3, self.gf_dim * 2,
                               kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.gf_dim * 2 * 4, self.gf_dim,
                               kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(self.gf_dim * 1 * 4, self.gf_dim,
                               kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(True)
        )

        self.final = nn.Sequential(
            PaddedConv2d(self.gf_dim, self.c_dim, ks=5, stride=1),
            nn.Tanh()
        )

    def forward(self, in_LDR, in_HDR):
        image1 = torch.cat([in_LDR[:, 0:self.c_dim, :, :], in_HDR[:, 0:self.c_dim, :, :]], 1)
        image2 = torch.cat([in_LDR[:, self.c_dim:self.c_dim * 2, :, :], in_HDR[:, self.c_dim:self.c_dim * 2, :, :]], 1)
        image3 = torch.cat([in_LDR[:, self.c_dim * 2:self.c_dim * 3, :, :], in_HDR[:, self.c_dim * 2:self.c_dim * 3, :, :]], 1)
        en1_t = self.encoder1_1(image1)
        en1 = self.encoder1_2(en1_t)
        en2_t = self.encoder2_1(image2)
        en2 = self.encoder2_2(en2_t)
        en3_t = self.encoder3_1(image3)
        en3 = self.encoder3_2(en3_t)
        en_all = torch.cat([en1, en2, en3], 1)
        merger_res = self.merger(en_all)
        residual_res = self.residual_blocks(merger_res)
        non_local_res = self.non_local(merger_res)
        res0 = torch.cat([merger_res, residual_res, non_local_res], 1)
        res1 = self.decoder1(res0)
        res1 = torch.cat([res1, en1, en2, en3], 1)
        res2 = self.decoder2(res1)
        res2 = torch.cat([res2, en1_t, en2_t, en3_t], 1)
        res3 = self.decoder3(res2)
        res = self.final(res3)
        return res
