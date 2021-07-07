"""
A rough translation of Magenta's Onsets and Frames implementation [1].
    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization

class MutliHeadAttention1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, groups=1, position=True, bias=False):
        """kernel_size is the 1D local attention window size"""

        super().__init__()
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.position = position
        
        # Padding should always be (kernel_size-1)/2
        # Isn't it?
        self.padding = (kernel_size-1)//2
        self.groups = groups

        # Make sure the feature dim is divisible by the n_heads
        assert self.out_features % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        assert (kernel_size-1) % 2 == 0, "kernal size must be odd number"

        if self.position:
            # Relative position encoding
            self.rel = nn.Parameter(torch.randn(1, out_features, kernel_size), requires_grad=True)

        # Input shape = (batch, len, feat)
        
        # Increasing the channel deapth (feature dim) with Conv2D
        # kernel_size=1 such that it expands only the feature dim
        # without affecting other dimensions
        self.W_k = nn.Linear(in_features, out_features, bias=bias)
        self.W_q = nn.Linear(in_features, out_features, bias=bias)
        self.W_v = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

    def forward(self, x):

        batch, seq_len, feat_dim = x.size()

        padded_x = F.pad(x, [0, 0, self.padding, self.padding])
        q_out = self.W_q(x)
        k_out = self.W_k(padded_x)
        v_out = self.W_v(padded_x)
        
        k_out = k_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        v_out = v_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        if self.position:
            k_out = k_out + self.rel # relative position?

        k_out = k_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)
        v_out = v_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)
        # (batch, L, n_heads, feature_per_head, local_window)
        
        # expand the last dimension s.t. it can multiple with the local att window
        q_out = q_out.view(batch, seq_len, self.groups, self.out_features // self.groups, 1)
        # (batch, L, n_heads, feature_per_head, 1)
        
        energy = (q_out * k_out).sum(-2, keepdim=True)
        
        attention = F.softmax(energy, dim=-1)
        # (batch, L, n_heads, 1, local_window)
        
        out = attention*v_out
#         out = torch.einsum('blnhk,blnhk -> blnh', attention, v_out).view(batch, seq_len, -1)
        # (batch, c, H, W)
        
        return out.sum(-1).flatten(2), attention.squeeze(3)

    def reset_parameters(self):
        init.kaiming_normal_(self.W_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_v.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_q.weight, mode='fan_out', nonlinearity='relu')
        if self.position:
            init.normal_(self.rel, 0, 1)
            
class MutliHeadAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), groups=1, bias=False):
        """kernel_size is the 2D local attention window size"""

        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Padding should always be (kernel_size-1)/2
        # Isn't it?
        self.padding_time = (kernel_size[0]-1)//2
        self.padding_freq = (kernel_size[1]-1)//2
        self.groups = groups

        # Make sure the feature dim is divisible by the n_heads
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # Relative position encoding
        self.rel_t = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size[0], 1), requires_grad=True)
        self.rel_f = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size[1]), requires_grad=True)

        # Increasing the channel deapth (feature dim) with Conv2D
        # kernel_size=1 such that it expands only the feature dim
        # without affecting other dimensions
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding_freq, self.padding_freq, self.padding_time, self.padding_time])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        
        k_out = k_out.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # (batch, channels, H, W, H_local_w, W_local_w) 
        
        v_out = v_out.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # (batch, channels, H, W, H_local_w, W_local_w) 

        k_out_t, k_out_f = k_out.split(self.out_channels // 2, dim=1)
        
        k_out = torch.cat((k_out_t + self.rel_t, k_out_f + self.rel_f), dim=1) # relative position?

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # (batch, n_heads, feature_per_head, H, W, local H X W)
        
        # expand the last dimension s.t. it can multiple with the local att window
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        # (batch, n_heads, feature_per_head, H, W, 1)

        # Alternative way to express dot product
        # same as k_out = k_out.permute(0,1,3,4,2,5)
        # and then energy = torch.matmul(q_out,k_out) 
        energy = (q_out * k_out).sum(dim=2, keepdim=True)
        
        attention = F.softmax(energy, dim=-1)
        # (batch, n_heads, 1, H, W, local HXW)
        
        out = attention*v_out
        # (batch, n_heads, feature_per_head, H, W, local HXW)
        # (batch, c, H, W)
        
        return out.sum(-1).flatten(1,2), attention.squeeze(2)

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_t, 0, 1)
        init.normal_(self.rel_f, 0, 1)          
        
class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, spec):
        x = spec.view(spec.size(0), 1, spec.size(1), spec.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class Onset_Stack(nn.Module):
    def __init__(self, input_features, model_size, output_features, sequence_model):
        super().__init__()
        self.convstack = ConvStack(input_features, model_size)
        self.sequence_model = sequence_model
        self.linear = nn.Linear(model_size, output_features)
    
    def forward(self, x):
        x = self.convstack(x)
        x, a = self.sequence_model(x)
        x = self.linear(x)
        
        return torch.sigmoid(x), a    
    
class Combine_Stack_with_attn(nn.Module):
    def __init__(self, model_size, output_features, sequence_model, attention_mode, w_size):
        super().__init__()
        self.sequence_model = sequence_model
        self.w_size = w_size    
        self.linear = nn.Linear(model_size, output_features)
        
    def forward(self, x):
        x, a = self.sequence_model(x)
        x = self.linear(x)
        
        return torch.sigmoid(x), a
    
    
class OnsetsAndFrames_self_attention(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', device='cpu', attention_mode='activation', w_size=30, n_heads=8, onset_stack=True, LSTM=True):
        super().__init__()
        self.onset_stack=onset_stack
        self.w_size=w_size
        self.device = device
        self.log = log
        self.normalize = Normalization(mode)        
        self.attention_mode=attention_mode
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: MutliHeadAttention1D(in_features=input_size,
                                                                              out_features=output_size,
                                                                              kernel_size=w_size,
                                                                              groups=n_heads)
        
        
         
        self.combined_stack = Combine_Stack_with_attn(model_size, output_features,
                                                      sequence_model(output_features * 2,
                                                                     model_size),
                                                      attention_mode,
                                                      w_size)
        self.onset_stack = Onset_Stack(input_features, model_size, output_features,
                                       sequence_model(model_size, model_size))

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    
    def forward(self, spec):
        onset_pred, onset_attention = self.onset_stack(spec)
        seq_len = onset_pred.shape[1]
#         offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(spec)

        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)

#         hidden = (h,c) # Setting the first hidden state to be the output from onset stack
        frame_pred, combined_attention = self.combined_stack(combined_pred) # Attenting on onset

        return onset_pred, activation_pred, frame_pred, combined_attention

       
        
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
#         offset_label = batch['offset']
        frame_label = batch['frame']
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        if self.onset_stack:
            predictions = {
#                 'onset': onset_pred.reshape(*onset_label.shape),
                'onset': onset_pred.reshape(*onset_label.shape),
    #             'offset': offset_pred.reshape(*offset_label.shape),
                'activation': activation_pred,
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a
    #             'velocity': velocity_pred.reshape(*velocity_label.shape)
            }
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
    #             'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
    #             'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            
        else:
            predictions = {
                'onset': frame_pred.reshape(*frame_label.shape),
                'activation': activation_pred,
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a
            }
            losses = {
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }            

        return predictions, losses, spec
    
    def feed_audio(self, audio):
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred,
#             'offset': offset_pred.reshape(*offset_label.shape),
            'activation': activation_pred,
            'frame': frame_pred,
            'attention': a
#             'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        return predictions, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class simple_onset_frame(nn.Module):           
    def __init__(self, input_features, output_features, model_complexity=48, w_size=31,
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        self.sequence_model_onset = MutliHeadAttention1D(in_features=input_features,
                                                          out_features=model_complexity,
                                                          kernel_size=w_size,
                                                          position=position,
                                                          groups=n_heads)
        self.layer_norm_onset = nn.LayerNorm(model_complexity)  
        self.linear_onset = nn.Linear(model_complexity, output_features)
        
        self.sequence_model_frame = MutliHeadAttention1D(in_features=model_complexity+output_features,
                                                          out_features=model_complexity,
                                                          kernel_size=w_size,
                                                          position=position,
                                                          groups=n_heads)
        self.layer_norm_frame = nn.LayerNorm(model_complexity)  
        self.linear_frame = nn.Linear(model_complexity, output_features)
        
        


     
        

    
    def forward(self, spec):
        x, a = self.sequence_model_onset(spec)
        x = self.layer_norm_onset(x)
        onset_pred = torch.sigmoid(self.linear_onset(x))
            
        # Version 1 try concate
        x = torch.cat((onset_pred, x), -1)
        # Version 2 try add
        x, _ = self.sequence_model_frame(x)   
        x = self.layer_norm_frame(x)
        frame_pred = torch.sigmoid(self.linear_frame(x))
        
        return frame_pred, onset_pred, a

       
        
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
  
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        frame_pred, onset_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred.reshape(*frame_label.shape),
#             'activation': frame_pred.reshape(*frame_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'attention': a
                      }
        
        if self.training:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }
        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }            
            
        return predictions, losses, spec
    
    def feed_audio(self, audio):
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred,
#             'offset': offset_pred.reshape(*offset_label.shape),
            'activation': activation_pred,
            'frame': frame_pred,
            'attention': a
#             'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        return predictions, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)      
            
class standalone_self_attention_1D(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, w_size=31,
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True, layernorm_pos=None):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.layernorm_pos = layernorm_pos
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        self.sequence_model = MutliHeadAttention1D(in_features=input_features,
                                              out_features=model_complexity,
                                              kernel_size=w_size,
                                              position=position,
                                              groups=n_heads)
        if layernorm_pos=='Before':
            self.layer_norm = nn.LayerNorm(model_complexity)  
        elif layernorm_pos=='After':
            self.layer_norm = nn.LayerNorm(output_features)             
        self.linear = nn.Linear(model_complexity, output_features)

    
    def forward(self, spec):
        x, a = self.sequence_model(spec)
        if self.layernorm_pos=='Before':
            x = self.layer_norm(x)
            
        x = self.linear(x)
            
        if self.layernorm_pos=='After':
            x = self.layer_norm(x)
            
        frame_pred = torch.sigmoid(x)
        
        return frame_pred, a

       
        
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
  
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        frame_pred, a = self(spec)
        
        predictions = {
            'onset': frame_pred.reshape(*frame_label.shape),
#             'activation': frame_pred.reshape(*frame_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'attention': a
                      }
        
        if self.training:
            losses = {
    #             'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }
        else:
            losses = {
    #             'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }            
            
        return predictions, losses, spec
    
    def feed_audio(self, audio):
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred,
#             'offset': offset_pred.reshape(*offset_label.shape),
            'activation': activation_pred,
            'frame': frame_pred,
            'attention': a
#             'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        return predictions, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)   
            
class standalone_self_attention_2D(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=16, w_size=(3,3),
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        self.sequence_model = MutliHeadAttention2D(in_channels=input_features,
                                                   out_channels=model_complexity,
                                                   kernel_size=w_size,
                                                   stride=(1,1),
                                                   groups=1, bias=False)
        

        self.linear = nn.Linear(N_BINS*model_complexity, output_features)


    
    def forward(self, spec):
        spec = spec.unsqueeze(1)
        x, a = self.sequence_model(spec)
        x = x.transpose(1,2).flatten(2)
        frame_pred = torch.sigmoid(self.linear(x))
        
        return frame_pred, a

       
        
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
  
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        frame_pred, a = self(spec)
        
        predictions = {
            'onset': frame_pred.reshape(*frame_label.shape),
#             'activation': frame_pred.reshape(*frame_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
#             'attention': a
                      }
        
        if self.training:
            losses = {
    #             'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }
        else:
            losses = {
    #             'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }            
            
        return predictions, losses, spec
    
    def feed_audio(self, audio):
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred,
#             'offset': offset_pred.reshape(*offset_label.shape),
            'activation': activation_pred,
            'frame': frame_pred,
            'attention': a
#             'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        return predictions, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)               