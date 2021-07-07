import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization


def _l2_normalize(d, binwise):
    # input shape (batch, timesteps, bins, ?)
    if binwise==True:
        d = d/(torch.abs(d)+1e-8)
    else:
        d = d/(torch.norm(d, dim=-1, keepdim=True))
    return d



class Seg_VAT(nn.Module):
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
            y_ref = model(x) # This will be used as a label, therefore no need grad()
            
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
            y_pred = model(x_adv)
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
        y_pred = model(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(y_pred, y_ref)          
        else:
            vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv, _l2_normalize(d, binwise=self.binwise)  # already averaged    

def calculate_padding(input_size, kernel_size, stride):
    def calculate_padding_1D(input_size, kernel_size, stride):
        if (input_size%stride==0):
            pad = max(kernel_size-stride, 0)
        else:
            pad = max(kernel_size-(input_size%stride), 0)

        return pad
    
    if type(kernel_size) != tuple:
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    else:
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]      

    if type(stride) != tuple:
        stride_1 = stride
        stride_2 = stride
    else:
        stride_1 = stride[0]
        stride_2 = stride[1]

    padding1 = calculate_padding_1D(input_size[0], kernel_size_1, stride_1)
    padding2 = calculate_padding_1D(input_size[1], kernel_size_2, stride_2)

    pad_top = padding1//2
    pad_bottom = padding1 - pad_top
    pad_left = padding2//2
    pad_right = padding2 - pad_left
    
    return (pad_left,pad_right,pad_top,pad_bottom)  

def transpose_padding_same(x, input_shape, stride):
    """
    Trying to implement padding='SAME' as in tensorflow for the Conv2dTranspose layer.
    It is basically trying to remove paddings from the output
    """
    
    input_shape = torch.tensor(input_shape[2:])*torch.tensor(stride)
    output_shape = torch.tensor(x.shape[2:])
    
    if torch.equal(input_shape,output_shape):
        print(f'same, no need to do anything')
        pass
    else:
        padding_remove = (output_shape-input_shape)
        left = padding_remove//2
        right = padding_remove//2+padding_remove%2    
        
    return x[:,:,left[0]:-right[0],left[1]:-right[1]]

def SAME_padding(x, ksize, stride):
    padding = calculate_padding(x.shape[2:], ksize, stride)
    return F.pad(x, padding) 


class Conv_Block(nn.Module):
    def __init__(self, inp, out, ksize, stride=(2,2), dilation_rate=1, dropout_rate=0.4):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.stride_conv2 = 1
        self.ksize_skip = 1
        
        padding=0 # We don't pad with the Conv2d class, we use F.pad to pad instead
        
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(inp)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, stride=self.stride_conv2, padding=padding, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out)
        self.dropout2 = nn.Dropout(dropout_rate)
        
       
        self.conv_skip = nn.Conv2d(inp, out, kernel_size=self.ksize_skip, stride=stride, padding=padding)
        

    def forward(self, x):
        skip = x # save a copy for the skip connection later
        
        x = self.bn1(torch.relu(x))
        x = self.dropout1(x)
        
        # Calculating padding corresponding to 'SAME' in tf
        x = SAME_padding(x, self.ksize, self.stride)
        x = self.conv1(x)
        
        x = self.bn2(torch.relu(x))
        x = self.dropout2(x)
        
        # Calculating padding corresponding to 'SAME' in tf
        x = SAME_padding(x, self.ksize, self.stride_conv2)
        x = self.conv2(x)
        
        if self.stride!=(1,1):
        # Calculating padding corresponding to 'SAME' in tf
            skip = SAME_padding(skip, self.ksize_skip, self.stride)
            # Padding is mostly 0 so far, comment it out first
            skip = self.conv_skip(skip)
        x = x + skip # skip connection
        
        return x


class transpose_conv_block(nn.Module):
    def __init__(self, inp, out, ksize, stride=(2,2), dropout_rate=0.4):
        super().__init__()
        
        self.stride = stride
        self.ksize = ksize
        padding=0 # We don't pad with the Conv2d class, we use F.pad to pad instead
        
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, stride=(1,1), padding=padding)
        self.bn1 = nn.BatchNorm2d(inp)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        
        self.conv2 = nn.ConvTranspose2d(out, out, kernel_size=ksize, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv_skip = nn.ConvTranspose2d(inp, out, kernel_size=1, stride=stride, padding=padding)
        

    def forward(self, x, shape):
        skip = x # save a copy for the skip connection later
        input_shape_skip = skip.shape # will be used as in the transpose padding later
        
        x = self.bn1(torch.relu(x))
        x = self.dropout1(x)
        x = SAME_padding(x, self.ksize, (1,1))
        x = self.conv1(x)
        
#         transpose_conv1 = torch.Size([1, 128, 40, 15])        
        
        x = self.bn2(torch.relu(x))
        x = self.dropout2(x)
        input_shape = x.shape
        x = self.conv2(x)
        x = transpose_padding_same(x, input_shape, self.stride)
        
        # Removing extra pixels induced due to ConvTranspose
        if x.shape[2]>shape[2]:
            x = x[:,:,:-1,:]
        if x.shape[3]>shape[3]:
            x = x[:,:,:,:-1]           
        
#         transpose_conv2 = torch.Size([1, 128, 83, 35])        
        
        if self.stride!=(1,1):
            # Check keras about the transConv output shape
            skip = self.conv_skip(skip, output_size=x.shape) # make output size same as x
#             skip = transpose_padding_same(skip, input_shape_skip, self.stride)
    
        x = x + skip # skip connection
        
        return x
    
class Decoder_Block(nn.Module):
    def __init__(self,
                 input_channels,
                 encoder_channels,
                 hidden_channels,
                 output_channels,
                 dropout_rate=0.4):
        super().__init__()
        
        # Again, not using Conv2d to calculate the padding,
        # use F.pad to obtain a more general padding under forward
        self.ksize = (1,1)
        self.stride = (1,1)
        self.layer1a = nn.Conv2d(input_channels+encoder_channels, hidden_channels, kernel_size=self.ksize, stride=self.stride) # the channel dim for feature
        self.bn = nn.BatchNorm2d(input_channels)
        self.bn_en = nn.BatchNorm2d(encoder_channels)
        self.dropout1 = nn.Dropout(dropout_rate)    
        self.layer1b = transpose_conv_block(input_channels, output_channels, (3,3), (2,2))
        

    def forward(self, x, encoder_output, encoder_shape):
        skip = x # save a copy for the skip connection later
        
        x = self.bn(torch.relu(x))

        en_l = self.bn_en(torch.relu(encoder_output))
        
        x = torch.cat((x, en_l), 1)
        x = self.dropout1(x)
        
        x = SAME_padding(x, self.ksize, self.stride)
        x = self.layer1a(x)
        x = x + skip
        
        x = self.layer1b(x, encoder_shape)
        
        return x
    
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

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 feature_num=128,
                 timesteps=256,
                 multi_grid_layer_n=1,
                 multi_grid_n=3,
                 ch_num=1,
                 prog=False,
                 dropout_rate=0.4,
                 out_class=2):
        super().__init__()
        
        # Parameters for the encoding layer
        en_kernel_size = (7,7)
        en_stride = (1,1)
        # Again, not using Conv2d to calculate the padding,
        # use F.pad to obtain a more general padding under forward
        self.en_padding = calculate_padding(input_size, en_kernel_size, en_stride)
        # Instead of using Z, it should be using Z_f and Z_q
        # But for the sake of this experiment, 
        self.encoding_layer = nn.Conv2d(1, 2**5, kernel_size=en_kernel_size, stride=en_stride, padding=0)

        self.layer1a = Conv_Block(2**5, 2**5, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer1b = Conv_Block(2**5, 2**5, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer2a = Conv_Block(2**5, 2**6, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer2b = Conv_Block(2**6, 2**6, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer2c = Conv_Block(2**6, 2**6, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer3a = Conv_Block(2**6, 2**7, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer3b = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer3c = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer3d = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer4a = Conv_Block(2**7, 2**8, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer4b = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4c = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4d = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4e = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)

        

    def forward(self, x):
        skip = x # save a copy for the skip connection later
        original_shape = x.shape
        
        x = F.pad(x, self.en_padding)
        x = self.encoding_layer(x)
        x = self.layer1a(x)
        x = self.layer1b(x)
        en_l1 = x
        shape1 = x.shape
        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.layer2c(x)
        shape2 = x.shape
        en_l2 = x

        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = self.layer3d(x)
        shape3 = x.shape
        en_l3 = x

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        x = self.layer4d(x)
        x = self.layer4e(x)
        shape4 = x.shape
        en_l4 = x
        
        # en_l4 and shape4 could not be used inside the decoder, that's why they are omitted
        return x, (en_l1, en_l2, en_l3), (original_shape, shape1, shape2, shape3)
    
    
class Decoder(nn.Module):
    def __init__(self,
                 dropout_rate=0.4):
        super().__init__()
        
        self.de_layer1 = Decoder_Block(2**7, 2**7, 2**7, 2**6, dropout_rate)
        self.de_layer2 = Decoder_Block(2**6, 2**6, 2**6, 2**6, dropout_rate)
        self.de_layer3 = Decoder_Block(2**6, 2**5, 2**6, 2**6, dropout_rate)
        

    def forward(self, x, encoder_outputs, encoder_shapes):
        x = self.de_layer1(x, encoder_outputs[-1], encoder_shapes[-2])
        x = self.de_layer2(x, encoder_outputs[-2], encoder_shapes[-3])
        x = self.de_layer3(x, encoder_outputs[-3], encoder_shapes[-4]) # Check this
        return x
    
    
class Semantic_Segmentation(nn.Module):
    def __init__(self, x, out_class=2, dropout_rate=0.4, log=True,
                 mode='imagewise', spec='Mel', device='cpu', XI=1e-6, eps=1e-2):
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
        self.vat_loss = Seg_VAT(XI, eps, 1, False)
        
        self.encoder = Encoder((x.shape[2:]), dropout_rate=dropout_rate)
        self.attention_layer1 = MutliHeadAttention2D(256, 64, kernel_size=(17,17), stride=(1,1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention_layer2 = MutliHeadAttention2D(64, 128, kernel_size=(17,17), stride=(1,1), groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # L218-221 of the original code
        # Few layers before the Decoder part        
        self.layer0a = nn.Conv2d(384, 2**8, (1,1), (1,1))
        self.layer0b = transpose_conv_block(2**8, 2**7, (3,3), (2,2))
        
        self.decoder = Decoder(dropout_rate=dropout_rate)
        
        # Last few layers that determines the output
        self.bn_last = nn.BatchNorm2d(2**6)
        self.dropout_last = nn.Dropout(dropout_rate)
        self.conv_last = nn.Conv2d(2**6, out_class, (1,1), (1,1))
        
        self.inference_model = nn.Linear(x.shape[-1], 88)
        
    def forward(self, x):
        x, encoder_outputs, encoder_shapes = self.encoder(x)
        en_l4 = x # Will be appened with the attention output and decoder later

        # Two layers of self-attention 
        x,_ = self.attention_layer1(en_l4)
        x = self.bn1(torch.relu(x))

        x, _ = self.attention_layer2(x)
        x = self.bn2(torch.relu(x))
        x = torch.cat((en_l4, x),1) # L216

        # L218-221 of the original code
        # Few layers before the Decoder part
        x = SAME_padding(x, (1,1), (1,1))
        x = self.layer0a(x)
        x = x + en_l4   
        x = self.layer0b(x, encoder_shapes[-1]) # Transposing back to the Encoder shape
        
        # Decoder part
        x = self.decoder(x, encoder_outputs, encoder_shapes)   
        
        # Last few layers for the output block
        x = self.bn_last(torch.relu(x))
        x = self.dropout_last(x)
        x = self.conv_last(x)
        
        # We use a Linear layer as the inference model here
        x = x.squeeze(1) # remove the channel dim
        x = self.inference_model(x)
        x = torch.sigmoid(x)
        
        
        return x
    
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
        

        frame_pred = self(spec)
        if self.training:
            predictions = {
                    'onset': frame_pred,
                    'frame': frame_pred,
                    'r_adv': r_adv,                
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
                    }                        

            losses = {
                    'loss/test_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                    'loss/test_LDS_l': lds_l,
                    'loss/test_r_norm_l': r_norm_l.abs().mean()                  
                    }                            

        return predictions, losses, spec.squeeze(1)
    
    def transcribe(self, batch):
        audio_label = batch['audio']
        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)      
        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)
        
        pianoroll = self(spec)     
        
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