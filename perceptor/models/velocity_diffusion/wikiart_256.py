import math
import torch
from torch import nn


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def alpha_sigma_to_log_snr(alpha, sigma):
    """Returns a log snr, given the scaling factors for the clean image and for
    the noise."""
    return torch.log(alpha**2 / sigma**2)


def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return log_snr.sigmoid().sqrt(), log_snr.neg().sigmoid().sqrt()


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -torch.special.expm1(1e-4 + 10 * ddpm_t**2).log()
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.skip = skip if skip else nn.Identity()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv2d(c_in, c_mid, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, padding=1),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1):
        super().__init__()
        assert c_in % n_head == 0
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(input)
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.skip(input), self.main(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class WikiArt256Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 256, 256)
        self.min_t = get_ddpm_schedule(torch.tensor(0.0)).item()
        self.max_t = get_ddpm_schedule(torch.tensor(1.0)).item()

        c = 128  # The base channel count
        cs = [c // 2, c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.net = nn.Sequential(  # 256x256
            ResConvBlock(3 + 16, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,  # 256x256 -> 128x128
                    ResConvBlock(cs[0], cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,  # 128x128 -> 64x64
                            ResConvBlock(cs[1], cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,  # 64x64 -> 32x32
                                    ResConvBlock(cs[2], cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,  # 32x32 -> 16x16
                                            ResConvBlock(cs[3], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            SkipBlock(
                                                [
                                                    self.down,  # 16x16 -> 8x8
                                                    ResConvBlock(cs[4], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    SkipBlock(
                                                        [
                                                            self.down,  # 8x8 -> 4x4
                                                            ResConvBlock(
                                                                cs[5], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[6]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6], cs[6] // 128
                                                            ),
                                                            ResConvBlock(
                                                                cs[6], cs[6], cs[5]
                                                            ),
                                                            SelfAttention2d(
                                                                cs[5], cs[5] // 128
                                                            ),
                                                            self.up,  # 4x4 -> 8x8
                                                        ]
                                                    ),
                                                    ResConvBlock(
                                                        cs[5] * 2, cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 128
                                                    ),
                                                    ResConvBlock(cs[5], cs[5], cs[4]),
                                                    SelfAttention2d(
                                                        cs[4], cs[4] // 128
                                                    ),
                                                    self.up,  # 8x8 -> 16x16
                                                ]
                                            ),
                                            ResConvBlock(cs[4] * 2, cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 128),
                                            ResConvBlock(cs[4], cs[4], cs[3]),
                                            SelfAttention2d(cs[3], cs[3] // 128),
                                            self.up,  # 16x16 -> 32x32
                                        ]
                                    ),
                                    ResConvBlock(cs[3] * 2, cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[3]),
                                    ResConvBlock(cs[3], cs[3], cs[2]),
                                    self.up,  # 32x32 -> 64x64
                                ]
                            ),
                            ResConvBlock(cs[2] * 2, cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[2]),
                            ResConvBlock(cs[2], cs[2], cs[1]),
                            self.up,  # 64x64 -> 128x128
                        ]
                    ),
                    ResConvBlock(cs[1] * 2, cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[1]),
                    ResConvBlock(cs[1], cs[1], cs[0]),
                    self.up,  # 128x128 -> 256x256
                ]
            ),
            ResConvBlock(cs[0] * 2, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], 3, is_last=True),
        )

    def forward(self, input, t):
        log_snr = alpha_sigma_to_log_snr(*t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(
            self.timestep_embed(log_snr[:, None]), input.shape
        )
        return self.net(torch.cat([input, timestep_embed], dim=1))
