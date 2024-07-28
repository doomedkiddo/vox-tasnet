import torch
import torch.nn as nn

class InputModule(nn.Module):
    def __init__(self, num_channels, sample_rate, window_size, window_stride):
        super(InputModule, self).__init__()
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.window_size = window_size  # In seconds
        self.window_stride = window_stride  # In seconds
        
        # Convert window size and stride to samples
        self.window_length = int(window_size * sample_rate)
        self.stride_length = int(window_stride * sample_rate)
        
        # Define the initial 2D convolution layer to reduce time resolution
        self.encoder_conv = nn.Conv2d(in_channels=num_channels,
                                      out_channels=64,
                                      kernel_size=(1, 64),
                                      stride=(1, 32),
                                      padding=(0, (64 - 32) // 2))  # Assuming some padding to maintain width

    def forward(self, x):
        # Assuming x is a batch of stereo audio signals with shape (batch_size, num_channels, audio_length)
        # Reshape x to (batch_size, num_channels, 1, audio_length) to make it suitable for 2D convolution
        x = x.unsqueeze(2)  # Add a dummy dimension for the 2D convolution
        
        # Apply the 2D convolutional layer to reduce time resolution
        encoded = self.encoder_conv(x)
        
        # Squeeze the dummy dimension back
        encoded = encoded.squeeze(2)
        
        # The output of the encoder is now ready to be fed into the separator module
        return encoded

# Example usage:
# Define the input module with the appropriate parameters
input_module = InputModule(num_channels=2,  # Stereo audio
                            sample_rate=44100,  # 44.1 kHz sample rate
                            window_size=0.01,   # 10 ms window size
                            window_stride=0.005) # 5 ms window stride

# Create a random batch of stereo audio data
batch_size = 16
audio_length = 4 * 44100  # 4 seconds of audio at 44.1 kHz
input_audio = torch.randn(batch_size, 2, audio_length)  # (batch_size, num_channels, audio_length)

# Pass the input audio through the input module
encoded_audio = input_module(input_audio)
print(encoded_audio)
