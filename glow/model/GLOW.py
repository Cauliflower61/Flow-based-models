import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    # initialize mean and std in minibatch #
    def Initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)       # shape: [1, channels, 1, 1]
            std = torch.std(x, dim=[0, 2, 3], keepdim=True)        # shape: [1, channels, 1, 1]

            self.mean.data.copy_(mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        # [B, C, H, W] -> [B, C, H, W]
        _, _, height, width = x.shape
        if self.initialized.item() == 0:
            self.Initialize(x)
            self.initialized.fill_(1)
        # calculate log_determinant #
        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        return (x - self.mean) * self.scale, logdet

    def inverse(self, x):
        return x / self.scale + self.mean


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        Q, R = torch.linalg.qr(torch.randn(in_channel, in_channel))
        P, L, U = torch.linalg.lu(Q)
        s = torch.diag(U)

        U_mask = torch.triu(torch.ones_like(U), 1)
        L_mask = U_mask.T

        self.register_buffer("P", P)
        self.register_buffer("U_mask", U_mask)
        self.register_buffer("L_mask", L_mask)
        self.register_buffer("s_sign", torch.sign(s))
        self.register_buffer("L_eye", torch.eye(L_mask.shape[0]))

        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.s = nn.Parameter(logabs(s))

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.P
            @ (self.L * self.L_mask + self.L_eye)
            @ ((self.U * self.U_mask) + torch.diag(self.s_sign * torch.exp(self.s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def inverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class AddCoupling(nn.Module):
    def __init__(self, in_channels, mid_channels=512):
        super().__init__()
        half_channels = int(in_channels // 2)
        self.f = nn.Sequential(nn.Conv2d(half_channels, mid_channels, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(mid_channels, mid_channels, 1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(mid_channels, half_channels, kernel_size=3, padding=1),
                               )
        self.scale = nn.Parameter(torch.zeros(size=(1, half_channels, 1, 1)))
        # normal initialized #
        self.f[0].weight.data.normal_(0, 0.05)
        self.f[0].bias.data.zero_()
        self.f[2].weight.data.normal_(0, 0.05)
        self.f[2].bias.data.zero_()
        # zero initialized #
        self.f[4].weight.data.zero_()
        self.f[4].bias.data.zero_()

    def forward(self, x):
        # y1 = x1 & y2 = x2 + f(x1)
        # input: [B, C, H, W] -> [B, C, H, W]
        part1, part2 = x.chunk(2, 1)
        f_out = -self.f(part1) * torch.exp(self.scale) + part2
        return torch.cat([part1, f_out], dim=1)

    def inverse(self, input):
        # x1 = y1 & x2 = y2 - f(x1)
        part1, part2 = input.chunk(2, 1)
        f_out = self.f(part1) * torch.exp(self.scale) + part2
        return torch.cat([part1, f_out], dim=1)


class Flow(nn.Module):
    # Flow = ActNorm + InvConv2d + AddCoupling
    def __init__(self, in_channels):
        super().__init__()
        self.ActNorm = ActNorm(in_channels)
        self.Shuffle = InvConv2d(in_channels)
        self.AddCoupling = AddCoupling(in_channels)

    def forward(self, x):
        # input: [B, C, H, W] -> [B, C, H, W]
        out, logdet = self.ActNorm(x)
        out, logdet1= self.Shuffle(out)
        out = self.AddCoupling(out)
        logdet += logdet1
        return out, logdet

    def inverse(self, x):
        out = self.AddCoupling.inverse(x)
        out = self.Shuffle.inverse(out)
        out = self.ActNorm.inverse(out)
        return out


def squeeze(x):
    # input: [B, C, H, W] -> [B, C*4, H/2, W/2]
    batch, channel, height, width = x.shape
    out = x.reshape(batch, channel, height // 2, 2, width // 2, 2)\
        .permute(0, 1, 3, 5, 2, 4).reshape(batch, channel*4, height//2, width//2)
    return out


def unsqueeze(x):
    # input: [B, 4*C, H/2, W/2] -> [B, C, H, W]
    batch, channel, height, width = x.shape
    out = x.reshape(batch, channel // 4, 2, 2, height, width)\
        .permute(0, 1, 4, 2, 5, 3).reshape(batch, channel//4, height*2, width*2)
    return out


def log_gaussian_likelihood(x, mean, log_std):
    return -0.5 * torch.log(2 * pi + torch.exp(2 * log_std)) - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


class Split(nn.Module):

    def __init__(self, in_channels, isSplit=True):
        super().__init__()
        self.isSplit = isSplit
        self.conv = nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1) if isSplit \
            else nn.Conv2d(in_channels*4, in_channels*8, kernel_size=3, padding=1)

        self.scale = nn.Parameter(torch.zeros(1, in_channels*4 if isSplit else in_channels*8, 1, 1))
        # zero initialized #
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        # input: [B, 4*C, H/2, W/2] -> Z: [B, 2*C, H/2, W/2]
        if self.isSplit:
            z1, z2 = x.chunk(2, 1)
            out = self.scale * self.conv(z2)
            mean, log_std = out.chunk(2, 1)
            log_p = torch.sum(torch.mean(log_gaussian_likelihood(z1, mean, log_std), dim=0))
            return (z1, z2), log_p
        # input: [B, 4*C, H/2, W/2] -> Z: [B, 2*C, H/2, W/2]
        else:
            z1 = x
            z2 = torch.zeros_like(z1)
            out = self.scale * self.conv(z2)
            mean, log_std = out.chunk(2, 1)
            log_p = torch.sum(torch.mean(log_gaussian_likelihood(z1, mean, log_std), dim=0))
            return z1, log_p

    def inverse(self, x):
        if self.isSplit:
            z1, z2 = x
            out = self.scale * self.conv(z2)
            mean, log_std = out.chunk(2, 1)
            z1 = z1 * torch.exp(log_std) + mean
            return torch.cat([z1, z2], dim=1)
        else:
            z1, _ = x
            z2 = torch.zeros_like(z1)
            out = self.scale * self.conv(z2)
            mean, log_std = out.chunk(2, 1)
            z1 = z1 * torch.exp(log_std) + mean
            return z1


class Block(nn.Module):
    # Block = squeeze + Flow + Split
    # squeeze: [B, C, H, W] -> [B, 4*C, H/2, W/2]
    # Flow: [B, 4*C, H/2, W/2] -> [B, 4*C, H/2, W/2]
    # Split: [B, 4*C, H/2, W/2] -> [B, 2*C, H/2, W/2] if isSplit else [B, 4*C, H/2, W/2]
    def __init__(self, num_flow, in_channels, isSplit=True):
        super().__init__()
        self.layer = nn.ModuleList()
        for cnt in range(num_flow):
            self.layer.append(Flow(4*in_channels))
        self.isSplit = isSplit
        self.split = Split(in_channels, self.isSplit)

    def forward(self, x):
        _, _, height, width = x.shape
        out = squeeze(x)
        logdet = 0
        for layer in self.layer:
            out, logdet1 = layer(out)
            logdet += logdet1
        out, log_p = self.split(out)

        return out, logdet, log_p

    def inverse(self, x):
        out = self.split.inverse(x)
        for layer in list(reversed(self.layer)):
            out = layer.inverse(out)
        out = unsqueeze(out)
        return out


class Glow(nn.Module):
    def __init__(self, in_channels, num_flow, num_block):
        super().__init__()
        self.num_block = num_block
        self.blocks = nn.ModuleList()
        for cnt in range(num_block - 1):
            self.blocks.append(Block(num_flow, in_channels, isSplit=True))
            in_channels *= 2
        self.blocks.append(Block(num_flow, in_channels, isSplit=False))

    def forward(self, x):
        log_det = 0
        log_p = 0
        z_list = []
        for block in self.blocks:
            out, log_det1, log_p1 = block(x)
            if block.isSplit:
                z1, z2 = out
                x = z2
            else:
                z1 = out
            log_det += log_det1
            log_p += log_p1
            z_list.append(z1)
        return z_list, log_det, log_p

    def inverse(self, x):
        out = x[self.num_block - 1]
        z = torch.zeros_like(out)
        for iter1, block in enumerate(list(reversed(self.blocks))):
            out = (x[self.num_block - iter1 - 1], z)
            z = block.inverse(out)
        return z

# loss function definition #
    def loss_fn(self, logdet, log_p):
        loss = -(log_p + logdet)
        return loss/(32*32*3), logdet/(32*32*3), log_p/(32*32*3)

    def calculate_z_shape(self, image_size, inc, sample_size):
        z_shape = []

        for cnt in range(self.num_block - 1):
            inc *= 2
            image_size //= 2
            z_shape.append((sample_size, inc, image_size, image_size))

        z_shape.append((sample_size, inc * 4, image_size // 2, image_size // 2))
        return z_shape

# gaussian sampling & generate images #
    def sample(self, inc, image_size, sample_size, sigma):
        z_shape = Glow.calculate_z_shape(self, image_size, inc, sample_size)
        z_list = []
        for shape in z_shape:
            z_list.append(sigma * torch.randn(size=shape).to('cuda'))
        x = Glow.inverse(self, z_list)
        return x


if __name__ == "__main__":
    net = Glow(in_channels=3, num_flow=32, num_block=4)
    image = torch.randn(size=(16, 3, 64, 64))
    z_list, logdet = net(image)



