"""
A rough translation of Magenta's Onsets and Frames implementation [1].
    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init


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

# class stepwise_VAT(nn.Module):
#     """
#     We define a function of regularization, specifically VAT.
#     """

#     def __init__(self, XI, epsilon, n_power, VAT_mode):
#         super().__init__()
#         self.n_power = n_power
#         self.XI = XI
#         self.epsilon = epsilon
#         self.VAT_mode = VAT_mode

#     def forward(self, model, x):  
#         with torch.no_grad():
#             onset_ref, activation_ref, frame_ref = model(x) # This will be used as a label, therefore no need grad()
            
#         # generate_virtual_adversarial_perturbation
#         d = torch.randn_like(x, requires_grad=True) # Need gradient
#         for _ in range(self.n_power):
#             r = self.XI * _l2_normalize(d)
#             onset_pred, activation_pred, frame_pred = model(x + r)
            
#             dist_onset =F.binary_cross_entropy(onset_pred, onset_ref)
#             dist_activation =F.binary_cross_entropy(activation_pred, activation_ref)
#             dist_frame =F.binary_cross_entropy(frame_pred, frame_ref)
            
#             if self.VAT_mode == 'onset':
#                 dist = dist_onset
#             elif self.VAT_mode == 'activation':
#                 dist = dist_activation
#             elif self.VAT_mode == 'frame':
#                 dist = dist_frame      
#             elif self.VAT_mode == 'all':
#                 dist = dist_frame + dist_activation + dist_onset                    
            
#             dist.backward() # Calculate gradient wrt d
#             d = d.grad.detach()
#             model.zero_grad() # prevent gradient change in the model    

#         # generating virtual labels and calculate VAT    
#         r_adv = self.epsilon * _l2_normalize(d)
#         onset_pred, activation_pred, frame_pred = model(x + r_adv)
        
#         vat_onset =F.binary_cross_entropy(onset_pred, onset_ref)
#         vat_activation =F.binary_cross_entropy(activation_pred, activation_ref)
#         vat_frame =F.binary_cross_entropy(frame_pred, frame_ref)

#         if self.VAT_mode == 'onset':
#             vat_loss = vat_onset
#         elif self.VAT_mode == 'activation':
#             vat_loss = vat_activation
#         elif self.VAT_mode == 'frame':
#             vat_loss = vat_frame      
#         elif self.VAT_mode == 'all':
#             vat_loss = vat_frame + vat_activation + vat_onset           
            
#         return vat_loss, r_adv  # already averaged
def binary_kl_div(y_pred, y_ref):
    y_pred = torch.clamp(y_pred, 0, 0.9999) # prevent inf in kl_div
    y_ref = torch.clamp(y_ref, 0, 0.9999)
    q = torch.stack((y_pred, 1-y_pred), -1)
    p = torch.stack((y_ref, 1-y_ref), -1)    
    return F.kl_div(p.log(), q, reduction='batchmean') 

class stepwise_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power, KL_Div):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.KL_Div = KL_Div
        
        if KL_Div==True:
            self.binwise = False
        else:
            self.binwise = False

    def forward(self, model, x):  
        with torch.no_grad():
            onset_ref, activation_ref, frame_ref  = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, binwise=self.binwise)
            x_adv = (x + r).clamp(0,1)
            onset_pred, activation_pred, frame_pred  = model(x_adv)
            if self.KL_Div==True:
                loss = binary_kl_div(frame_pred, frame_ref)
            else:   
                loss =F.binary_cross_entropy(frame_pred, frame_ref)
            loss.backward() # Calculate gradient wrt d
            d = d.grad.detach()*1e10
            model.zero_grad() # prevent gradient change in the model 

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d, binwise=self.binwise)
        assert torch.isnan(r_adv).any()==False, "r_adv contains nan"
        assert torch.isinf(r_adv).any()==False, "r_adv contains nan"
              
#         logit_p = logit.detach()
        x_adv = (x + r_adv).clamp(0,1)
        onset_pred, activation_pred, frame_pred  = model(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(frame_pred, frame_ref)                      
        else:
            vat_loss = F.binary_cross_entropy(frame_pred, frame_ref)              
            
        return vat_loss, r_adv, _l2_normalize(d*1e8, binwise=self.binwise)  # already averaged
    
class stepwise_VAT_frame_stack(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power, VAT_mode):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.VAT_mode = VAT_mode

    def forward(self, model, x):  
        with torch.no_grad():
            activation_ref, frame_ref = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, False)
            x_adv = (x+r).clamp(0,1)
            activation_pred, frame_pred = model(x_adv)
            
            dist_activation = F.mse_loss(activation_pred, activation_ref)
            dist_frame = F.binary_cross_entropy(frame_pred, frame_ref)
            
            if self.VAT_mode == 'activation':
                dist = dist_activation
            elif self.VAT_mode == 'frame':
                dist = dist_frame      
            elif self.VAT_mode == 'all':
                dist = dist_frame + dist_activation  
            
            dist.backward() # Calculate gradient wrt d
            d = d.grad.detach()*1e20   
            model.zero_grad() # prevent gradient change in the model    

        # generating virtual labels and calculate VAT    
#         print(f'dist = {dist}')
#         print(f'd mean = {d.mean()}\std = {d.std()}')
#         print(f'd norm mean = {_l2_normalize(d).mean()}\tstd = {_l2_normalize(d).std()}')
        r_adv = self.epsilon * _l2_normalize(d, False)
        assert torch.isnan(r_adv).any()==False, "r_adv exploded, please debug tune down the XI for VAT"
        assert torch.isinf(r_adv).any()==False, "r_adv vanished, please debug tune up the XI for VAT"
        x_adv = (x+r_adv).clamp(0,1)    
        activation_pred, frame_pred  = model(x_adv)

        assert torch.isnan(activation_pred).any()==False, "activation_pred is nan, please debug"
        assert torch.isnan(frame_pred).any()==False, "frame_pred is nan, please debug"        
        
        vat_activation =F.mse_loss(activation_pred, activation_ref)
        vat_frame =F.binary_cross_entropy(frame_pred, frame_ref)

        if self.VAT_mode == 'activation':
            vat_loss = vat_activation
        elif self.VAT_mode == 'frame':
            vat_loss = vat_frame      
        elif self.VAT_mode == 'all':
            vat_loss = vat_frame + vat_activation           
            
        return vat_loss, r_adv  # already averaged
    
class stepwise_VAT_onset_stack(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power, VAT_mode):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.VAT_mode = VAT_mode

    def forward(self, model, x):  
        with torch.no_grad():
            onset_ref = model(x) # This will be used as a label, therefore no need grad()

        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d)
            onset_pred = model(x + r)

            dist = F.binary_cross_entropy(onset_pred, onset_ref)

            dist.backward() # Calculate gradient wrt d
            d = d.grad.detach()
            model.zero_grad() # prevent gradient change in the model    

        # generating virtual labels and calculate VAT
        r_adv = self.epsilon * _l2_normalize(d)
        assert torch.isnan(r_adv).any()==False, "r_adv exploded, please debug tune down the XI for VAT"
        assert torch.isinf(r_adv).any()==False, "r_adv vanished, please debug tune up the XI for VAT"
            
        onset_pred  = model(x + r_adv)

        assert torch.isnan(activation_pred).any()==False, "activation_pred is nan, please debug"
        assert torch.isnan(frame_pred).any()==False, "frame_pred is nan, please debug"        
        
        vat_loss = F.binary_cross_entropy(onset_pred, onset_ref)       
            
        return vat_loss, r_adv  # already averaged     
    
def _l2_normalize(d, binwise):
    # input shape (batch, timesteps, bins, ?)
    if binwise==True:
        d = d/(torch.abs(d)+1e-8)
    else:
        d = d/(torch.norm(d, dim=-1, keepdim=True))
    return d

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
        if self.sequence_model:
            self.linear = nn.Linear(model_size, output_features)
            self.forward = self.forward_LSTM
        else:
            self.linear = nn.Linear(model_size, output_features)
            self.forward = self.forward_noLSTM
        
    def forward_LSTM(self, x):
        x = self.convstack(x)
        if self.training:
            x, (h, c) = self.sequence_model(x)
            
        else:
            self.train()
            x, (h, c) = self.sequence_model(x)
            self.eval()
        x = self.linear(x)
        
        return torch.sigmoid(x)
    
    def forward_noLSTM(self, x):
        x = self.convstack(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)        
    
    
class Combine_Stack(nn.Module):
    def __init__(self, model_size, output_features, sequence_model):
        super().__init__()
        self.sequence_model = sequence_model
        if self.sequence_model:
            self.linear = nn.Linear(model_size, output_features)
            self.forward = self.forward_LSTM            
        else:
            self.linear = nn.Linear(output_features, output_features)
            self.forward = self.forward_noLSTM            
        
    def forward_LSTM(self, x):
        if self.training:
            x, _ = self.sequence_model(x)
        else:
            self.train()
            x, _ = self.sequence_model(x)
            self.eval()
        x = self.linear(x)
        
        return torch.sigmoid(x)

    def forward_noLSTM(self,x):
        x = self.linear(x)
        return torch.sigmoid(x)

    
class Frame_stack_VAT(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', XI=1e-5, eps=10, VAT_mode='all'):
        super().__init__()
        self.log = log
        self.normalize = Normalization(mode)        
        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: nn.LSTM(input_size, output_size // 2,  batch_first=True, bidirectional=True)

#         sequence_model = lambda input_size, output_size: MutliHeadAttention1D(input_size, output_size, 31, position=True, groups=4)

        self.vat_loss = stepwise_VAT_frame_stack(XI, eps,1, VAT_mode)

        self.combined_stack = Combine_Stack(model_size, output_features, sequence_model(output_features, model_size))
#         self.combined_stack = Combine_Stack(model_size, output_features, None)



        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, spec):
        activation_pred = self.frame_stack(spec)
        combined_pred = activation_pred
        frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
        
        return activation_pred, frame_pred 

    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        frame_label = batch['frame']
          
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        if batch_ul and VAT:
            audio_label_ul = batch_ul['audio']  
            spec_ul = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec_ul = torch.log(spec_ul + 1e-5)
            spec_ul = self.normalize.transform(spec_ul)
            spec_ul = spec.transpose(-1,-2)
            lds_ul, _ = self.vat_loss(self, spec_ul)
        else:
            lds_ul = torch.tensor(0.)
        
        if VAT:
            lds_l, r_adv = self.vat_loss(self, spec)
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
        
        activation_pred, frame_pred  = self(spec)        
        
        predictions = {
            'onset': frame_pred.reshape(*frame_pred.shape),
#             'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'r_adv': r_adv
        }        
        
        # no need 
        if self.training:
            losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/train_LDS': (lds_ul+lds_l)/2
            }
            
        else:
            losses = {
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/test_LDS': lds_l
            }            

        return predictions, losses, spec
    
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            
class Onset_stack_VAT(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', XI=1e-5, eps=10, VAT_mode='all'):
        super().__init__()
        self.log = log
        self.normalize = Normalization(mode)        
        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: nn.LSTM(input_size, output_size // 2,  batch_first=True, bidirectional=True)

        self.vat_loss = stepwise_VAT_onset_stack(XI, eps, 1, VAT_mode)

        self.onset_stack = Onset_Stack(input_features, model_size, output_features, sequence_model(model_size, model_size))


    def forward(self, spec):
        onset_pred = self.onset_stack(spec)
        
        return onset_pred

    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
          
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        if batch_ul and VAT:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec_ul = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2)
            lds_ul, _ = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0.)
        
        if VAT:
            lds_l, r_adv = self.vat_loss(self, spec)
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
        
        onset_pred  = self(spec)        
        accuracy = (onset_label == (onset_pred>0.5)).float().sum()/onset_label.flatten(0).shape[0] 
        
        predictions = {
            'onset': onset_pred.reshape(*onset_pred.shape),
            'r_adv': r_adv
        }        
        
        # no need 
        if self.training:
            losses = {
                'loss/train_onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'metric/train_accuracy': accuracy,
                'loss/train_LDS': torch.mean(torch.stack((lds_ul,lds_l)),dim=0)
            }
            
        else:
            losses = {
                'loss/test_onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'metric/test_accuracy': accuracy,
                'loss/test_LDS': lds_l
            }            

        return predictions, losses, spec
    
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)               
    
    
class OnsetsAndFrames_VAT_full(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', XI=1e-5, eps=10, VAT_mode='all'):
        super().__init__()
        self.log = log
        self.normalize = Normalization(mode)        
        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: nn.LSTM(input_size, output_size // 2,  batch_first=True, bidirectional=True)
        # Need to rewrite this part, since we are going to modify the LSTM
        self.vat_loss = stepwise_VAT(XI, eps,1, False)

        self.onset_stack = Onset_Stack(input_features, model_size, output_features, sequence_model(model_size, model_size))
        self.combined_stack = Combine_Stack(model_size, output_features, sequence_model(output_features * 2, model_size))

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, spec):
        onset_pred = self.onset_stack(spec)
#         offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(spec)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
        
        return onset_pred, activation_pred, frame_pred 

    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        if self.onset_stack:
            onset_label = batch['onset']
#         offset_label = batch['offset']
        frame_label = batch['frame']
#         velocity_label = batch['velocity']
        
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2)
            
#             print(f'run_batch label = {frame_label.shape}')            

            lds_ul, _, r_norm_ul = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0.)
            r_norm_ul = torch.tensor(0.)

        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)        
        
        if VAT:
            lds_l, r_adv, r_norm_l = self.vat_loss(self, spec)
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
            r_norm_l = torch.tensor(0.)        
            
        onset_pred, activation_pred, frame_pred  = self(spec)        

        
        if self.training:
            predictions = {
                'onset': onset_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'r_adv': r_adv,
                          }            
            
            losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/train_onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/train_LDS_l': lds_l,
                'loss/train_LDS_ul': lds_ul,
                'loss/train_r_norm_l': r_norm_l.abs().mean(),
                'loss/train_r_norm_ul': r_norm_ul.abs().mean()                
            }
        else:
            predictions = {
                'onset': onset_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'r_adv': r_adv,
                          }                        
            
            losses = {
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/test_onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/test_LDS_l': lds_l,
                'loss/test_r_norm_l': r_norm_l.abs().mean()                  
            }            
      
        return predictions, losses, spec         

        return predictions, losses, spec
    
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)




