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
from itertools import cycle

def create_triangular_cycle(start, end, period):
    triangle_a = torch.linspace(start,end,period)
    triangle_b = torch.linspace(end,start,period)[1:-1]
    triangle=torch.cat((triangle_a,triangle_b))
    
    return cycle(triangle)

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
        assert self.out_features % self.groups == 0, f"out_channels should be divided by groups. (example: out_channels: 40, groups: 4). Now out_channels={self.out_features}, groups={self.groups}"
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
            

class stepwise_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power, KL_Div, binwise=False):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.KL_Div = KL_Div
        
        self.binwise = binwise

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _ = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, binwise=self.binwise)
            x_adv = (x + r).clamp(0,1)
            y_pred, _ = model(x_adv)
            if self.KL_Div==True:
                loss = binary_kl_div(y_pred, y_ref)
            else:   
                loss =F.binary_cross_entropy(y_pred, y_ref)
            loss.backward() # Calculate gradient wrt d
            d = d.grad.detach()
            model.zero_grad() # prevent gradient change in the model 

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d, binwise=self.binwise)
        
#         logit_p = logit.detach()
        x_adv = (x + r_adv).clamp(0,1)
        y_pred, _ = model(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(y_pred, y_ref)          
        else:
            vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv, _l2_normalize(d, binwise=self.binwise)  # already averaged
    
class UNet_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power, KL_Div, reconstruction=False):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.KL_Div = KL_Div
        
        self.binwise = False
        self.reconstruction = reconstruction

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _ = model.transcriber(x) # This will be used as a label, therefore no need grad()
            
#             if self.reconstruction:
#                 pianoroll, _ = model.transcriber(x)
#                 reconstruction, _ = self.reconstructor(pianoroll)
#                 pianoroll2_ref, _ = self.transcriber(reconstruction)                
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
#         if self.reconstruction:
#             d2 = torch.randn_like(x, requires_grad=True) # Need gradient            
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, binwise=self.binwise)
            x_adv = (x + r).clamp(0,1)
            y_pred, _ = model.transcriber(x_adv)
            if self.KL_Div==True:
                loss = binary_kl_div(y_pred, y_ref)
            else:   
                loss =F.binary_cross_entropy(y_pred, y_ref)
            loss.backward() # Calculate gradient wrt d
            d = d.grad.detach()*1e10
            model.zero_grad() # prevent gradient change in the model 

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d, binwise=self.binwise)
        assert torch.isnan(r_adv).any()==False, f"r_adv has nan, d min={d.min()} d max={d.max()} d mean={d.mean()} please debug tune down the XI for VAT"
        assert torch.isnan(r_adv).any()==False, f"r_adv has inf, d min={d.min()} d max={d.max()} d mean={d.mean()} please debug tune down the XI for VAT"
#         print(f'd max = {d.max()}\td min = {d.min()}')
#         print(f'r_adv max = {r_adv.max()}\tr_adv min = {r_adv.min()}')        
#         logit_p = logit.detach()
        x_adv = (x + r_adv).clamp(0,1)
        y_pred, _ = model.transcriber(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(y_pred, y_ref)          
        else:
            vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv, _l2_normalize(d, binwise=self.binwise)  # already averaged    
    
class onset_frame_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _, _ = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, binwise=False)
            x_adv = (x + r).clamp(0,1)            
            y_pred, _, _ = model(x_adv)
            dist =F.binary_cross_entropy(y_pred, y_ref)
            dist.backward() # Calculate gradient wrt d
            d = d.grad.detach()
            model.zero_grad() # prevent gradient change in the model    

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d, binwise=False)
#         logit_p = logit.detach()
        x_adv = (x + r_adv).clamp(0,1)
        y_pred, _, _ = model(x_adv)
#         print(f'x_adv max = {x_adv.max()}\tx_adv min = {x_adv.min()}')
        vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv  # already averaged
    
def _l2_normalize(d, binwise):
    # input shape (batch, timesteps, bins, ?)
    if binwise==True:
        d = d/(torch.abs(d)+1e-8)
    else:
        d = d/(torch.norm(d, dim=-1, keepdim=True))
    return d

def binary_kl_div(y_pred, y_ref):
    y_pred = torch.clamp(y_pred, 1e-4, 0.9999) # prevent inf in kl_div
    y_ref = torch.clamp(y_ref, 1e-4, 0.9999)
    q = torch.stack((y_pred, 1-y_pred), -1)
    p = torch.stack((y_ref, 1-y_ref), -1) 
    assert torch.isnan(p.log()).any()==False, "r_adv exploded, please debug tune down the XI for VAT"
    assert torch.isinf(p.log()).any()==False, "r_adv vanished, please debug tune up the XI for VAT"
    return F.kl_div(p.log(), q, reduction='batchmean')   
            
class VAT_self_attention_1D(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, w_size=31,
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True, XI=1e-5, eps=1e-2,
                 eps_period=False, eps_max=1, KL_Div=False):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        self.sequence_model = MutliHeadAttention1D(in_features=input_features,
                                              out_features=model_complexity,
                                              kernel_size=w_size,
                                              position=position,
                                              groups=n_heads)

        self.layer_norm = nn.LayerNorm(model_complexity)             
        self.linear = nn.Linear(model_complexity, output_features)
        
        self.vat_loss = stepwise_VAT(XI, eps, 1, KL_Div, False)
        self.eps_period = eps_period
        if self.eps_period:
            self.triangular_cycle = create_triangular_cycle(eps,eps_max,eps_period)        

    
    def forward(self, spec):
        x, a = self.sequence_model(spec)
        x = self.layer_norm(x)
            
        x = self.linear(x)
            
        frame_pred = torch.sigmoid(x)
        
        return frame_pred, a

       
        
    def run_on_batch(self, batch_l, batch_ul=None, VAT=False):
        audio_label = batch_l['audio']
        onset_label = batch_l['onset']
        frame_label = batch_l['frame'] 
        
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2)

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
        
        frame_pred, a = self(spec)
        
        if self.training:
            if self.eps_period:            
                self.vat_loss.eps = next(self.triangular_cycle)   
            print(f'eps = {self.vat_loss.eps}')    
            predictions = {
                'onset': frame_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv,
                          }            
            
            losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/train_LDS_l': lds_l,
                'loss/train_LDS_ul': lds_ul,
                'loss/train_r_norm_l': r_norm_l.abs().mean(),
                'loss/train_r_norm_ul': r_norm_ul.abs().mean()                
            }
        else:
            predictions = {
                'onset': frame_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv,
                          }                        
            
            losses = {
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/test_LDS_l': lds_l,
                'loss/test_r_norm_l': r_norm_l.abs().mean()                  
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
            
class ConvStack(nn.Module):
    def __init__(self, output_features):
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
        
        freq_features = self._get_conv_output()
        self.fc = nn.Sequential(
            nn.Linear(freq_features, output_features),
            nn.Dropout(0.5)
        )
    def forward(self, spec):
        x = self.cnn(spec)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x      
    
    def _get_conv_output(self):
        shape = (1, 640, 229)
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
#         n_size = output_feat.data.view(bs, -1).size(1)
        return output_feat.transpose(1, 2).flatten(-2).size(-1)
    
    def _forward_features(self, x):
        x = self.cnn(x)
        return x      
    

class Timbral_CNN(nn.Module):
    def __init__(self, start_channel, final_channel, output_features):
        super().__init__()    
        self.cnn = nn.Sequential(
            # Old
#                             nn.Conv2d(1, start_channel, (1, 21)),
#                             nn.BatchNorm2d(start_channel),
#                             nn.ReLU(),
            
# #                             nn.Conv2d(start_channel, start_channel, (1, 51)),
# #                             nn.ReLU(),
#                             nn.Conv2d(start_channel, start_channel, (3, 51), padding=(1,0)),
#                             nn.BatchNorm2d(start_channel),            
#                             nn.ReLU(),
#                             nn.MaxPool2d((1, 2)),
# #                             nn.Dropout(0.25),
# #                             nn.Conv2d(start_channel//2, start_channel//2, (3, 51), padding=(1,0)),
# #                             nn.ReLU(),    
#                             nn.Conv2d(start_channel, final_channel, (7, 21), padding=(3,0)),
#                             nn.BatchNorm2d(final_channel),            
#                             nn.ReLU(),
#                             nn.MaxPool2d((1, 2)),
# #                             nn.Dropout(0.25),
# #                             nn.Conv2d(final_channel, final_channel, (7, 51), padding=(3,0)),
# #                             nn.ReLU(), 
            # new------------
                            nn.Conv2d(1, start_channel, (3, 3), padding=1),
                            nn.BatchNorm2d(start_channel),
                            nn.ReLU(),
            
#                             nn.Conv2d(start_channel, start_channel, (1, 51)),
#                             nn.ReLU(),
                            nn.Conv2d(start_channel, start_channel, (3, 3), padding=1),
                            nn.BatchNorm2d(start_channel),            
                            nn.ReLU(),
                            nn.MaxPool2d((1, 2)),
#                             nn.Dropout(0.25),
#                             nn.Conv2d(start_channel//2, start_channel//2, (3, 51), padding=(1,0)),
#                             nn.ReLU(),    
                            nn.Conv2d(start_channel, final_channel, (3, 3), padding=1),
                            nn.BatchNorm2d(final_channel),            
                            nn.ReLU(),
                            nn.MaxPool2d((1, 2)),
#                             nn.Dropout(0.25),
#                             nn.Conv2d(final_channel, final_channel, (7, 51), padding=(3,0)),
#                             nn.ReLU(),                
        )
        # input is batch_size * 1 channel * frames * input_features
        freq_features = self._get_conv_output()
        self.fc = nn.Sequential(
            nn.Linear(freq_features, output_features),
        )
    def forward(self, spec):
        x = self.cnn(spec)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x          
    def _get_conv_output(self):
        shape = (1, 640, 229)
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
#         n_size = output_feat.data.view(bs, -1).size(1)
        return output_feat.transpose(1, 2).flatten(-2).size(-1)
    
    def _forward_features(self, x):
        x = self.cnn(x) 
        return x              

 
            
class VAT_CNN_attention_1D(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, w_size=31,
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True, XI=1e-5, eps=1e-2, version='a'):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)



        if version=='a':
            self.cnn = ConvStack(output_features)

        elif version=='b':
#         input is batch_size * 1 channel * frames * input_features
            self.cnn = Timbral_CNN(32,8,output_features)

        self.sequence_model = MutliHeadAttention1D(in_features=output_features,
                                              out_features=model_complexity,
                                              kernel_size=w_size,
                                              position=position,
                                              groups=n_heads)    

        self.layer_norm = nn.LayerNorm(model_complexity)             
        self.linear = nn.Linear(model_complexity, output_features)
        
        self.vat_loss = stepwise_VAT(XI, eps,1, False)
        
        self.triangular_cycle = create_triangular_cycle(1e-2,10,50)

#     def _get_conv_output(self):
#         shape = (1, 640, 229)
#         bs = 1
#         input = torch.rand(bs, *shape)
#         output_feat = self._forward_features(input)
# #         n_size = output_feat.data.view(bs, -1).size(1)
#         return output_feat.size(-1)
    
#     def _forward_features(self, x):
#         x = self.cnn(x)
#         return x            
    
    def forward(self, spec):
        x = self.cnn(spec.unsqueeze(1))
#         x = x.transpose(1,2).flatten(2)
        # 1 Layer = ([8, 8, 640, 687])
        x, a = self.sequence_model(x)
        x = self.layer_norm(x)
            
        x = self.linear(x)
            
        frame_pred = torch.sigmoid(x)
        
        return frame_pred, a

       
        
    def run_on_batch(self, batch_l, batch_ul=None, VAT=False):
        audio_label = batch_l['audio']
        onset_label = batch_l['onset']
        frame_label = batch_l['frame'] 
        
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2)

            lds_ul, _ = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0)
         
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        if VAT:
            lds_l, r_adv = self.vat_loss(self, spec)          
        else:
            r_adv = None
            lds_l = torch.tensor(0)
        frame_pred, a = self(spec)
        self.vat_loss.eps = next(self.triangular_cycle)
        print(f'VAT eps={self.vat_loss.eps}')
        
#         print(f'loss = {F.binary_cross_entropy(frame_pred, frame_label)}')
        if self.training:
            predictions = {
                'onset': frame_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv
                          }            
            
            losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/train_LDS_l': lds_l,
                'loss/train_LDS_ul': lds_ul
            }
        else:
            predictions = {
                'onset': frame_pred.reshape(*frame_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv
                          }                        
            
            losses = {
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
    
class VAT_CNN_attention_onset_frame(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, w_size=31,
                 log=True, mode='imagewise', spec='Mel', n_heads=8, position=True, XI=1e-5, eps=1e-2):
        super().__init__()
        self.w_size=w_size
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        start_channel = 48
        final_channel = 96
        # input is batch_size * 1 channel * frames * input_features
        self.cnn = Timbral_CNN(start_channel,final_channel,output_features)
        self.onset_timbral_cnn = Timbral_CNN(start_channel, final_channel, output_features)     
        
        freq_features = self._get_conv_output()
        self.onset_attention = MutliHeadAttention1D(in_features=output_features,
                                              out_features=model_complexity,
                                              kernel_size=w_size,
                                              position=position,
                                              groups=n_heads)
        self.layer_norm_onset = nn.LayerNorm(model_complexity)
        self.onset_classifier = nn.Linear(model_complexity, output_features)

        
        self.final_attention = MutliHeadAttention1D(in_features=2*output_features,
                                              out_features=model_complexity,
                                              kernel_size=w_size,
                                              position=position,
                                              groups=n_heads)
        self.layer_norm_final = nn.LayerNorm(model_complexity)             
        self.final_classifier = nn.Linear(model_complexity, output_features)
        
        self.vat_loss = onset_frame_VAT(XI, eps,1)

    def _get_conv_output(self):
        shape = (1, 640, 229)
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
#         n_size = output_feat.data.view(bs, -1).size(1)
        return output_feat.size(-1)
    
    def _forward_features(self, x):
        x = self.cnn(x)
        return x            
    
    def forward(self, spec):
        onset_pred = self.onset_timbral_cnn(spec.unsqueeze(1))
        onset_pred, _ = self.onset_attention(onset_pred)

#         onset_pred, _ = self.onset_attention(spec)
        onset_pred = self.layer_norm_onset(onset_pred)
        onset_pred = self.onset_classifier(onset_pred)
        onset_pred = torch.sigmoid(onset_pred)
        
        activation = self.cnn(spec.unsqueeze(1))
        # activation shape = (8, 8, 640, freq_freatures)
        # activation shape = (8, 640, freq_freatures*8)
        
        # 1 Layer = ([8, 8, 640, 687])
        x, a = self.final_attention(torch.cat((onset_pred, activation), dim=-1))
        x = self.layer_norm_final(x)
        x = self.final_classifier(x)
        frame_pred = torch.sigmoid(x)
        
        return frame_pred, onset_pred, a

       
        
    def run_on_batch(self, batch_l, batch_ul=None, VAT=False):
        audio_label = batch_l['audio']
        onset_label = batch_l['onset']
        frame_label = batch_l['frame'] 
        
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2)

            lds_ul, _ = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0.)
         
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        if VAT:
            lds_l, r_adv = self.vat_loss(self, spec)
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
        
        frame_pred, onset_pred, a = self(spec)
        
#         print(f'loss = {F.binary_cross_entropy(frame_pred, frame_label)}')
        if self.training:
            predictions = {
                'onset': onset_pred.reshape(*onset_pred.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv
                          }            
            
            losses = {
                'loss/train_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/train_onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/train_LDS_l': lds_l,
                'loss/train_LDS_ul': lds_ul
            }
        else:
            predictions = {
                'onset': onset_pred.reshape(*onset_pred.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a,
                'r_adv': r_adv
                          }                        
            
            losses = {
                'loss/test_frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/test_onset': F.binary_cross_entropy(predictions['onset'], onset_label),                
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

batchNorm_momentum = 0.1
num_instruments = 1


class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 

    def forward(self, x, size=None, isLast=None, skip=None):
        # print(f'x.shape={x.shape}')
        # print(f'target shape = {size}')
        x = self.us(x,output_size=size)
        if not isLast: x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Encoder, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

    def forward(self, x):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
       
        c1=self.conv1(x3) 
        c2=self.conv2(x2) 
        c3=self.conv3(x1) 
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]

class Decoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Decoder, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block4 = d_block(16,num_instruments,True,(3,3),(1,1),ds_ksize, ds_stride)
            

            
    def forward(self, x, s, c=[None,None,None,None]):
        x = self.d_block1(x,s[3],False,c[0])
        x = self.d_block2(x,s[2],False,c[1])
        x = self.d_block3(x,s[1],False,c[2])
        x = self.d_block4(x,s[0],True,c[3])
#         reconsturction = torch.sigmoid(self.d_block4(x,s[0],True,c[3]))
       
#         return torch.sigmoid(x) # This is required to boost the accuracy
        return x # This is required to boost the accuracy

    
class Spec2Roll(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
        self.lstm1 = MutliHeadAttention1D(N_BINS, N_BINS*complexity, 31, position=True, groups=complexity)
        # self.lstm1 = nn.LSTM(N_BINS, N_BINS, batch_first=True, bidirectional=True)    
        self.linear1 = nn.Linear(N_BINS*complexity, 88)     
        
    def forward(self, x):
        # U-net 1
        x,s,c = self.Unet1_encoder(x)
        x = self.Unet1_decoder(x,s,c)
        x, a = self.lstm1(x.squeeze(1)) # remove the channel dim
        pianoroll = torch.sigmoid(self.linear1(x)) # Use the full LSTM output
        
        return pianoroll, a
    
class Roll2Spec(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        self.Unet2_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet2_decoder = Decoder(ds_ksize, ds_stride)   
#             self.lstm2 = nn.LSTM(88, N_BINS, batch_first=True, bidirectional=True)
        self.lstm2 = MutliHeadAttention1D(88, N_BINS*complexity, 31, position=True, groups=4)            
        self.linear2 = nn.Linear(N_BINS*complexity, N_BINS)         
        
    def forward(self, x):
#         U-net 2
        x, a = self.lstm2(x)
        x= torch.sigmoid(self.linear2(x)) # ToDo, remove the sigmoid activation and see if we get a better result
        x,s,c = self.Unet2_encoder(x.unsqueeze(1))
        reconstruction = self.Unet2_decoder(x,s,c) # predict roll

#         x,s,c = self.Unet2_encoder(x.unsqueeze(1))
#         x = self.Unet2_decoder(x,s,c) # predict roll
#         x, a = self.lstm2(x.squeeze(1))
#         reconstruction = self.linear2(x) # ToDo, remove the sigmoid activation and see if we get a better result        
#         reconstruction = reconstruction.clamp(0,1).unsqueeze(1)
        
        return reconstruction, a

class Reconstructor(nn.Module):
    def __init__(self, ds_ksize, ds_stride):
        super().__init__() 
        self.reconstructor = Roll2Spec(ds_ksize, ds_stride)     
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                              hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                              trainable_mel=False, trainable_STFT=False)
        self.normalize = Normalization('imagewise')
        
    def forward(self, x):
        reconstruction, a = self.reconstructor(x)
        
        return reconstruction, a    
    
    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        

        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)
    
        
        reconstrut, a = self(frame_label)           
        predictions = {
                'attention': a,               
                'reconstruction': reconstrut,
                }
        losses = {
                'loss/train_reconstruction': F.binary_cross_entropy(reconstrut.squeeze(1), spec.squeeze(1).detach()),
                }                  

        return predictions, losses, spec.squeeze(1)

        
class UNet(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, reconstruction=True, mode='imagewise', spec='CQT', device='cpu', XI=1e-6, eps=1e-2):
        super().__init__()
        global N_BINS # using the N_BINS parameter from constant.py
        
        # Selecting the type of spectrogram to use
        if spec == 'CQT':
            r=2
            N_BINS = 88*r
            self.spectrogram = Spectrogram.CQT1992v2(sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                      n_bins=N_BINS, fmin=27.5, 
                                                      bins_per_octave=12*r, trainable=False)
        elif spec == 'Mel':
            self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
        elif spec == 'CFP':
            self.spectrogram = Spectrogram.CFP(fs=SAMPLE_RATE,
                                               fr=4,
                                               window_size=WINDOW_LENGTH,
                                               hop_length=HOP_LENGTH,
                                               fc=MEL_FMIN,
                                               tc=1/MEL_FMAX)       
            N_BINS = self.spectrogram.quef2logfreq_matrix.shape[0]
        else:
            print(f'Please select a correct spectrogram')                

        self.log = log
        self.normalize = Normalization(mode)          
        self.reconstruction = reconstruction    
        self.vat_loss = UNet_VAT(XI, eps, 1, False)        
            
#         self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
#         self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
#         self.lstm1 = MutliHeadAttention1D(N_BINS, N_BINS*4, 31, position=True, groups=4)
#         # self.lstm1 = nn.LSTM(N_BINS, N_BINS, batch_first=True, bidirectional=True)    
#         self.linear1 = nn.Linear(N_BINS*4, 88)      
        self.transcriber = Spec2Roll(ds_ksize, ds_stride)

        if reconstruction==True:
#             self.Unet2_encoder = Encoder(ds_ksize, ds_stride)
#             self.Unet2_decoder = Decoder(ds_ksize, ds_stride)   
# #             self.lstm2 = nn.LSTM(88, N_BINS, batch_first=True, bidirectional=True)
#             self.lstm2 = MutliHeadAttention1D(88, N_BINS*4, 31, position=True, groups=4)            
#             self.linear2 = nn.Linear(N_BINS*4, N_BINS)      
            self.reconstructor = Roll2Spec(ds_ksize, ds_stride)
               
    def forward(self, x):

        # U-net 1
        pianoroll, a = self.transcriber(x)

        if self.reconstruction:     
            # U-net 2
            reconstruction, a_reconstruct = self.reconstructor(pianoroll)
            # Applying U-net 1 to the reconstructed spectrograms
            pianoroll2, a_2 = self.transcriber(reconstruction)
            
#             # U-net2
#             x, h = self.lstm2(pianoroll)
#             feat2= torch.sigmoid(self.linear2(x)) # ToDo, remove the sigmoid activation and see if we get a better result
#             x,s,c = self.Unet2_encoder(feat2.unsqueeze(1))
#             reconstruction = self.Unet2_decoder(x,s,c) # predict roll

#             # Applying U-net 1 to the reconstructed spectrograms
#             x,s,c = self.Unet1_encoder(reconstruction)
#             feat1b = self.Unet1_decoder(x,s,c)
#             x, h = self.lstm1(feat1b.squeeze(1)) # remove the channel dim
#             pianoroll2 = torch.sigmoid(self.linear1(x)) # Use the full LSTM output

            return reconstruction, pianoroll, pianoroll2, a
        else:
            return pianoroll, a



    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
            
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2).unsqueeze(1)

            lds_ul, _, r_norm_ul = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0.)
            r_norm_ul = torch.tensor(0.)

        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)      
        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)

        if VAT:
            lds_l, r_adv, r_norm_l = self.vat_loss(self, spec)
            r_adv = r_adv.squeeze(1) # remove the channel dimension
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
            r_norm_l = torch.tensor(0.)        
        
        if self.reconstruction:
            reconstrut, pianoroll, pianoroll2, a = self(spec)
            if self.training:             
                predictions = {
                    'onset': pianoroll,
                    'frame': pianoroll,
                    'frame2':pianoroll2,
                    'onset2':pianoroll2,
                    'attention': a,  
                    'r_adv': r_adv,                
                    'reconstruction': reconstrut,
                    }
                losses = {
                        'loss/train_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),
                        'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/train_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
                        'loss/train_LDS_l': lds_l,
                        'loss/train_LDS_ul': lds_ul,
                        'loss/train_r_norm_l': r_norm_l.abs().mean(),
                        'loss/train_r_norm_ul': r_norm_ul.abs().mean()                     
                        }
            else:
                predictions = {
                        'onset': pianoroll.reshape(*frame_label.shape),
                        'frame': pianoroll.reshape(*frame_label.shape),
                        'frame2':pianoroll2.reshape(*frame_label.shape),
                        'onset2':pianoroll2.reshape(*frame_label.shape),
                        'attention': a,   
                        'r_adv': r_adv,                
                        'reconstruction': reconstrut,                    
                        }                        
                
                losses = {
                        'loss/test_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),                 
                        'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/test_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
                        'loss/test_LDS_l': lds_l,
                        'loss/test_r_norm_l': r_norm_l.abs().mean()             
                        }                           

            return predictions, losses, spec.squeeze(1)

        else:
            frame_pred, a = self(spec)
            if self.training:
                predictions = {
                        'onset': frame_pred,
                        'frame': frame_pred,
                        'r_adv': r_adv,
                        'attention': a,                    
                        }
                losses = {
                        'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/train_LDS_l': lds_l,
                        'loss/train_LDS_ul': lds_ul,
                        'loss/train_r_norm_l': r_norm_l.abs().mean(),
                        'loss/train_r_norm_ul': r_norm_ul.abs().mean()                 
                        }
            else:
                predictions = {
                        'onset': frame_pred.reshape(*frame_label.shape),
                        'frame': frame_pred.reshape(*frame_label.shape),
                        'r_adv': r_adv,
                        'attention': a,                    
                        }                        
                
                losses = {
                        'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/test_LDS_l': lds_l,
                        'loss/test_r_norm_l': r_norm_l.abs().mean()                  
                        }                            

            return predictions, losses, spec.squeeze(1)
        
    def run_on_batch_application(self, batch, batch_ul=None, VAT=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
            
        if batch_ul:
            audio_label_ul = batch_ul['audio']  
            spec = self.spectrogram(audio_label_ul.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
            if self.log:
                spec = torch.log(spec + 1e-5)
            spec = self.normalize.transform(spec)
            spec = spec.transpose(-1,-2).unsqueeze(1)

            lds_ul, _, r_norm_ul = self.vat_loss(self, spec)
        else:
            lds_ul = torch.tensor(0.)
            r_norm_ul = torch.tensor(0.)
        reconstrut, ul_pianoroll, ul_pianoroll2, a = self(spec)   

        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)      
        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)

        if VAT:
            lds_l, r_adv, r_norm_l = self.vat_loss(self, spec)
            r_adv = r_adv.squeeze(1) # remove the channel dimension
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
            r_norm_l = torch.tensor(0.)        
        

        reconstrut, pianoroll, pianoroll2, a = self(spec)
        if self.training:             
            predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'frame2':pianoroll2,
                'onset2':pianoroll2,
                'ul_frame': ul_pianoroll,
                'ul_frame2': ul_pianoroll2,
                'attention': a,  
                'r_adv': r_adv,                
                'reconstruction': reconstrut,
                }
            losses = {
                    'loss/train_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),
                    'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                    'loss/train_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
#                     'loss/ul_consistency_wrt2': F.binary_cross_entropy(predictions['ul_frame'].squeeze(1), predictions['ul_frame2'].squeeze(1).detach()),
                    'loss/ul_consistency_wrt1': F.binary_cross_entropy(predictions['ul_frame2'].squeeze(1), predictions['ul_frame'].squeeze(1).detach()),                
                    'loss/train_LDS_l': lds_l,
                    'loss/train_LDS_ul': lds_ul,
                    'loss/train_r_norm_l': r_norm_l.abs().mean(),
                    'loss/train_r_norm_ul': r_norm_ul.abs().mean()                     
                    }
        else:
            predictions = {
                    'onset': pianoroll.reshape(*frame_label.shape),
                    'frame': pianoroll.reshape(*frame_label.shape),
                    'frame2':pianoroll2.reshape(*frame_label.shape),
                    'onset2':pianoroll2.reshape(*frame_label.shape),
                    'attention': a,   
                    'r_adv': r_adv,                
                    'reconstruction': reconstrut,                    
                    }                        

            losses = {
                    'loss/test_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),                 
                    'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                    'loss/test_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
                    'loss/test_LDS_l': lds_l,
                    'loss/test_r_norm_l': r_norm_l.abs().mean()             
                    }                           

        return predictions, losses, spec.squeeze(1)
    
    def transcribe(self, audio):
        audio
        # Converting audio to spectrograms
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)      
        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)
        
        reconstrut, pianoroll, pianoroll2, a = self(spec)     
        
        predictions = {
            'onset': pianoroll,
            'frame': pianoroll,         
            }
        
        return predictions

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
            