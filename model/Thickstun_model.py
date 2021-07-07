import torch
from torch.nn.functional import conv1d, mse_loss
import torch.nn.functional as F
import torch.nn as nn
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization

class Thickstun(torch.nn.Module):
    def __init__(self):
        super(Thickstun, self).__init__()      
        # Create filter windows
        # Creating Layers
        self.normalize = Normalization('imagewise')
        k_out = 128
        k2_out = 4096
        self.CNN_freq = nn.Conv2d(1,k_out,
                                kernel_size=(128,1),stride=(2,1))
        self.CNN_time = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,25),stride=(1,1))        
        self.linear = torch.nn.Linear(k2_out*51, 88, bias=False)
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        # Initialize weights
            # Do something
        
    def forward(self,x):
        
        z2 = torch.relu(self.CNN_freq(x.unsqueeze(1))) # Make channel as 1 (N,C,H,W) shape = [10, 128, 193, 25]
#         print(f'z2 = {z2.shape}')
        z3 = torch.relu(self.CNN_time(z2)) # shape = [10, 256, 193, 1]
#         print(f'z3 = {z3.shape}')        
        y = self.linear(torch.relu(torch.flatten(z3,1)))
        return torch.sigmoid(y)
    
    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        
        
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
            

        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)      
        # log compression
        spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # Change the shape such that it fits Thickstun Model
        
        spec_padded = torch.nn.functional.pad(spec, (12, 12)) # (batch, 229, 640+24)
        spec_padded = spec_padded.unfold(2, 25, 1) # extract 25 timesteps from the padded spec, stride=1, dim=2
        spec_padded = spec_padded.transpose(1,2).reshape(-1, 229, 25) # Cut spectrogram into segments as a batch   
       
        frame_pred = self(spec_padded)
#         print(f'output shape = {frame_pred.shape}')
#         print(f'label shape = {frame_label.shape}')
        predictions = {
                'onset': frame_pred,
                'frame': frame_pred,   
                'r_adv': None
                }
        losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label.reshape(-1,88)),
                }             

        return predictions, losses, spec.squeeze(1)