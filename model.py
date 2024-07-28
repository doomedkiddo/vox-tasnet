import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class cLN(nn.Module):
    def __init__(self, dimension):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, dimension, 1))
        self.beta = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-8) + self.beta

class S_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(S_Conv, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dconv = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               groups=out_channels, dilation=dilation, padding='same')
        self.prelu = nn.PReLU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.dconv(x)
        x = self.prelu(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(1, 384, kernel_size=(1, 64), stride=(1, 32))
        self.cln = cLN(768)

    def forward(self, x):
        # x shape: [B, 2, T] -> [B, 1, 2, T]
        x = x.unsqueeze(1)
        logger.info(x.size())
        x = self.conv2d(x)  # [B, 384, 2, T/32]
        logger.info(x.size())
        x = x.transpose(1, 2)  # [B, 2, 384, T/32]
        logger.info(x.size())

        # Apply cLN
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [B, 2*384, T/32]
        logger.info(x.size())
        x = self.cln(x)
        logger.info(x.size())

        # Reshape back
        x = x.reshape(x.shape[0], 2, 384, -1)  # [B, 2, 384, T/32]

        # Concatenate along the channel dimension
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [B, 2*384, T/32]

        return x

class VoxTasNet(nn.Module):
    def __init__(self):
        super(VoxTasNet, self).__init__()
        self.encoder = Encoder()
        self.separator = Separator()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)  # [B, 768, T/32]
        mask = self.separator(encoded)  # [B, 2, T/32]
        masked = encoded.unsqueeze(1) * mask.unsqueeze(2)  # [B, 2, 768, T/32]
        return self.decoder(masked)

# Modify the Separator class to accept 768 input channels
class Separator(nn.Module):
    def __init__(self):
        super(Separator, self).__init__()
        self.s_conv_layers = nn.ModuleList([
            S_Conv(768, 768, 3, 1),
            S_Conv(768, 768, 3, 2),
            S_Conv(768, 768, 3, 4),
            S_Conv(768, 768, 3, 8),
            S_Conv(768, 768, 3, 16),
            S_Conv(768, 768, 3, 32),
            S_Conv(768, 768, 3, 64),
            S_Conv(768, 768, 3, 128),
            S_Conv(768, 768, 3, 256),
        ])
        self.conv2d = nn.Conv2d(768, 2, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.s_conv_layers:
            x = layer(x)
        x = x.unsqueeze(2)  # [B, 768, 1, T/32]
        x = self.conv2d(x)  # [B, 2, 1, T/32]
        return self.sigmoid(x).squeeze(2)  # [B, 2, T/32]

# Modify the Decoder class to accept 768 input channels
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(768, 1, kernel_size=(1, 64), stride=(1, 32))

    def forward(self, x):
        # x shape: [B, 2, 768, T/32]
        x = x.transpose(1, 2)  # [B, 768, 2, T/32]
        x = self.transposed_conv(x)  # [B, 1, 2, T]
        return x.squeeze(1).transpose(1, 2)  # [B, 2, T]

# Example usage
if __name__ == "__main__":
    model = VoxTasNet()
    print(model)
    x = torch.randn(1, 2, 44100*4)  # 4 seconds of stereo audio at 44.1kHz
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
