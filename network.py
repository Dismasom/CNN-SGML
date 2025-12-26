import torch
from torch import nn, optim
from torch.nn import functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class UNET_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels=None, num_layers=8, width_multipliers=None, dropout=0.2):
        """
        可配置U-Net with CBAM
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param base_channels: 基础通道数
        :param num_layers: 网络总层数
        :param width_multipliers: 各层通道数倍增系数列表
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.cbam_modules = nn.ModuleList()

        # 初始化宽度系数
        if width_multipliers is None:
            width_multipliers = [min(2 ** (i), 8) for i in range(num_layers)]  # 限制最大8倍
        self.width_multipliers = width_multipliers

        # 构建编码器
        prev_ch = in_ch
        for i in range(num_layers):
            out_ch_en = base_channels * width_multipliers[i]
            encoder = nn.Sequential(
                nn.Conv2d(prev_ch, out_ch_en,kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True) if i > 0 else nn.Identity(),
                nn.BatchNorm2d(out_ch_en) if i > 0 else nn.Identity()
            )
            self.encoder_convs.append(encoder)
            self.cbam_modules.append(CBAM(out_ch_en))
            prev_ch = out_ch_en

        # 中间层
        self.mid_conv = nn.Sequential(
                nn.Conv2d(prev_ch, prev_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True) ,
                nn.BatchNorm2d(prev_ch)
            )

        # 构建解码器
        for i in reversed(range(num_layers)):
            if i==num_layers-1:
                in_ch_de=base_channels * width_multipliers[i]
                out_ch_de=in_ch_de
            else:
                in_ch_de = base_channels * width_multipliers[i+1] * 2  # 包含skip connection
                out_ch_de = base_channels * (width_multipliers[i] if i > 0 else 1)
            decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch_de, out_ch_de, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch_de),
                nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2, True)
            )
            self.decoder_convs.append(decoder)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(base_channels * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x)
            if i < len(self.cbam_modules):
                x = self.cbam_modules[i](x) + x
            skip_connections.append(x)

        # Middle layer
        x = self.mid_conv(x)

        # Decoder
        for i, conv in enumerate(self.decoder_convs):
            x = conv(x)
            # 跳跃连接拼接
            x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)  # -2开始取对应层

        # Final output
        x = self.final_conv(x)

        return x
    

class SpectralConv2d(nn.Module):
    """2D Fourier layer: 在频域中对前若干模式做线性变换"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 高度方向保留的频率数
        self.modes2 = modes2  # 宽度方向保留的频率数

        scale = 1.0 / (in_channels * out_channels)
        # 复权重，作用在低频子块上
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weight):
        # input: [B, in_c, H, W/2+1], weight: [in_c, out_c, modes1, modes2]
        return torch.einsum("bixy,ioxy->boxy", input, weight)

    def forward(self, x):
        batchsize, _, height, width = x.shape
        # rfft2 只在宽度方向输出一半频率（实数优化）
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        h_modes = min(self.modes1, height)
        w_modes = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :h_modes, :w_modes] = self.compl_mul2d(
            x_ft[:, :, :h_modes, :w_modes], self.weight[:, :, :h_modes, :w_modes]
        )

        x = torch.fft.irfft2(out_ft, s=(height, width))
        return x
       
class FNO2D(nn.Module):
    """2D FNO 结构：输入为 256x256 mask（或多通道场），输出对应流场

    输入尺寸: [B, in_ch, 256, 256]
    输出尺寸: [B, out_ch, 256, 256]
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        modes1: int = 32,
        modes2: int = 32,
        width: int = 64,
        depth: int = 4,
        use_tanh: bool = True,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.use_tanh = use_tanh

        # 线性升维: in_ch → width
        self.fc0 = nn.Conv2d(in_ch, width, kernel_size=1)

        # 若干个 FNO Block
        self.spectral_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        for _ in range(depth):

            self.spectral_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.pointwise_layers.append(nn.Conv2d(width, width, kernel_size=1))

        # 线性降维到目标通道
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_ch*8, kernel_size=1)
        self.fc3 = nn.Conv2d(out_ch*8, out_ch*8, kernel_size=3, padding=1)
        self.fc4 = nn.Conv2d(out_ch*8, out_ch, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x: [B, in_ch, 256, 256]
        x = self.fc0(x)

        for spec_conv, pw_conv in zip(self.spectral_layers, self.pointwise_layers):
            y1 = spec_conv(x)
            y2 = pw_conv(x)
            x = self.activation(y1 + y2)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        if self.use_tanh:
            x = torch.tanh(x)
        return x

