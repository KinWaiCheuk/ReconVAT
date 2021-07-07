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
            frame_ref, onset_ref, _ = model.transcriber(x) # This will be used as a label, therefore no need grad()
            
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
            frame_pred, onset_pred, _ = model.transcriber(x_adv)
            if self.KL_Div==True:
                loss = binary_kl_div(y_pred, y_ref)
            else:   
                frame_loss = F.binary_cross_entropy(frame_pred, frame_ref)
                onset_loss = F.binary_cross_entropy(onset_pred, onset_ref)
                
            loss = (frame_loss + onset_loss)
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
        frame_pred, onset_pred, _ = model.transcriber(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(y_pred, y_ref)          
        else:
            vat_frame_loss = F.binary_cross_entropy(frame_pred, frame_ref)              
            vat_onset_loss = F.binary_cross_entropy(onset_pred, onset_ref) 
            
        vat_loss = {'frame': vat_frame_loss,
                    'onset': vat_onset_loss}        
        return vat_loss, r_adv, _l2_normalize(d, binwise=self.binwise)  # already averaged    
    
    
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
                   

batchNorm_momentum = 0.1


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
    def __init__(self,ds_ksize, ds_stride, num_instruments):
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

class Stack(nn.Module):
    def __init__(self, input_size, hidden_dim, attn_size=31, attn_group=4, output_dim=88, dropout=0.5):
        super().__init__() 
        self.attention = MutliHeadAttention1D(input_size, hidden_dim, attn_size, position=True, groups=attn_group)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, x):
        x, a = self.attention(x)
        x = self.linear(x)
        x = self.dropout(x)

        return x, a
    
class Spec2Roll(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride, 2)
        self.lstm1 = MutliHeadAttention1D(N_BINS+88, N_BINS*complexity, 31, position=True, groups=complexity)
        # self.lstm1 = nn.LSTM(N_BINS, N_BINS, batch_first=True, bidirectional=True)    
        self.linear1 = nn.Linear(N_BINS*complexity, 88)
        self.linear_onset = nn.Linear(N_BINS, 88)
        self.linear_feature = nn.Linear(N_BINS, 88)
        self.dropout_layer = nn.Dropout(0.5)
                                      
#         self.onset_stack = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=4, output_dim=88, dropout=0)
        
        
#         self.feat_stack = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=4, output_dim=88, dropout=0)
        
        self.combine_stack = Stack(input_size=88*2, hidden_dim=768, attn_size=31, attn_group=6, output_dim=88, dropout=0)
        
    def forward(self, x):
        # U-net 1
        x,s,c = self.Unet1_encoder(x)
        x = self.Unet1_decoder(x,s,c)
        onset = self.linear_onset(x[:,0])
        onset = torch.sigmoid(onset)
        
        feat = self.linear_feature(x[:,1])
        x = torch.cat((onset, feat), -1)
        x, a = self.combine_stack(x)
        pianoroll = torch.sigmoid(x)
        
        return pianoroll, onset, a
    
class Roll2Spec(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        self.Unet2_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet2_decoder = Decoder(ds_ksize, ds_stride, 1)   
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
        
class UNet_Onset(nn.Module):
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
        pianoroll, onset, a = self.transcriber(x)

        if self.reconstruction:     
            # U-net 2
            reconstruction, a_reconstruct = self.reconstructor(pianoroll)
            # Applying U-net 1 to the reconstructed spectrograms
            pianoroll2, onset2, a_2 = self.transcriber(reconstruction)
            
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

            return reconstruction, pianoroll, onset, pianoroll2, onset2, a
        else:
            return pianoroll, onset, a



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
            lds_ul = {'frame': torch.tensor(0.),
                     'onset': torch.tensor(0.)}
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
            lds_l = {'frame': torch.tensor(0.),
                     'onset': torch.tensor(0.)}
            r_norm_l = torch.tensor(0.)        
        
        
        if frame_label.dim()==2:
            frame_label=frame_label.unsqueeze(0)
        if onset_label.dim()==2:
            onset_label=onset_label.unsqueeze(0)        
        
        if self.reconstruction:
            reconstrut, pianoroll, onset, pianoroll2, onset2, a = self(spec)
            if self.training:             
                predictions = {
                        'frame': pianoroll,
                        'onset': onset,
                        'frame2':pianoroll2,
                        'onset2':onset2,
                        'attention': a,  
                        'r_adv': r_adv,                
                        'reconstruction': reconstrut,
                    }
                losses = {
                        'loss/train_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),
                        'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/train_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
                        'loss/train_onset': F.binary_cross_entropy(predictions['onset'].squeeze(1), onset_label),
                        'loss/train_onset2': F.binary_cross_entropy(predictions['onset2'].squeeze(1), onset_label),               
                        'loss/train_LDS_l_frame': lds_l['frame'],
                        'loss/train_LDS_l_onset': lds_l['onset'],
                        'loss/train_LDS_ul_frame': lds_ul['frame'],
                        'loss/train_LDS_ul_onset': lds_ul['onset'],
                        'loss/train_r_norm_l': r_norm_l.abs().mean(),
                        'loss/train_r_norm_ul': r_norm_ul.abs().mean()                     
                        }
            else:
                predictions = {
                        'frame': pianoroll.reshape(*frame_label.shape),
                        'onset': onset.reshape(*onset.shape),
                        'frame2':pianoroll2.reshape(*frame_label.shape),
                        'onset2':onset2.reshape(*onset.shape),
                        'attention': a,   
                        'r_adv': r_adv,                
                        'reconstruction': reconstrut,                    
                        }                        
                
                losses = {
                        'loss/test_reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.squeeze(1).detach()),                 
                        'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/test_frame2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label),
                        'loss/test_onset': F.binary_cross_entropy(predictions['onset'].squeeze(1), onset_label),
                        'loss/test_onset2': F.binary_cross_entropy(predictions['onset2'].squeeze(1), onset_label),
                        'loss/test_LDS_l_frame': lds_l['frame'],
                        'loss/test_LDS_l_onset': lds_l['onset'],
                        'loss/test_r_norm_l': r_norm_l.abs().mean()             
                        }                           

            return predictions, losses, spec.squeeze(1)

        else:
            frame_pred, onset, a = self(spec)
            if self.training:
                predictions = {
                        'onset': onset,
                        'frame': frame_pred,
                        'r_adv': r_adv,
                        'attention': a,                    
                        }
                losses = {
                        'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/train_onset': F.binary_cross_entropy(predictions['onset'].squeeze(1), onset_label),
                        'loss/train_LDS_l_frame': lds_l['frame'],
                        'loss/train_LDS_l_onset': lds_l['onset'],
                        'loss/train_LDS_ul_frame': lds_ul['frame'],
                        'loss/train_LDS_ul_onset': lds_ul['onset'],
                        'loss/train_r_norm_l': r_norm_l.abs().mean(),
                        'loss/train_r_norm_ul': r_norm_ul.abs().mean()                 
                        }
            else:
                predictions = {
                        'onset': onset.reshape(*onset.shape),
                        'frame': frame_pred.reshape(*frame_label.shape),
                        'r_adv': r_adv,
                        'attention': a,                    
                        }                        
                losses = {
                        'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                        'loss/test_onset': F.binary_cross_entropy(predictions['onset'].squeeze(1), onset_label),
                        'loss/test_LDS_l_frame': lds_l['frame'],
                        'loss/test_LDS_l_onset': lds_l['onset'],
                        'loss/test_r_norm_l': r_norm_l.abs().mean()                  
                        }                            

            return predictions, losses, spec.squeeze(1)
        
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameterds
                param = param.data
            own_state[name].copy_(param)
            