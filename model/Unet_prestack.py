import torch
from torch.nn.functional import conv1d, mse_loss
import torch.nn.functional as F
import torch.nn as nn
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization


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

    
class Prestack(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
        
    def forward(self, x):
        # U-net 1
        x,s,c = self.Unet1_encoder(x)
        x = self.Unet1_decoder(x,s,c)
        
        return x
    


class Prestack_Model(nn.Module):
    def __init__(self, model='resnet18'):
        super().__init__()           
        unet = Prestack((3,3),(1,1))
        resnet = torch.hub.load('pytorch/vision:v0.9.0', model, pretrained=False)
        resnet.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        resnet.fc = torch.nn.Linear(512, 88, bias=True)                 
        self.prestack_model = nn.Sequential(unet, resnet)
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)    
        self.normalize = Normalization('imagewise')
                 
    def forward(self, x):
        return self.prestack_model(x)
                 
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
        spec_padded = spec_padded.unsqueeze(1) # create 1 channel for CNN   
#         print(f'spec_padded shape = {spec_padded.shape}')
        frame_pred = torch.zeros(spec_padded.shape[0], 88).to(spec_padded.device)
        for idx, i in enumerate(spec_padded):
            output = self(i.unsqueeze(0)).squeeze(0)     
            frame_pred[idx] = output
#             print(f'idx = {idx}\tfoward done = {output.shape}')
        frame_pred = torch.sigmoid(frame_pred)
            
#         frame_pred = torch.sigmoid(self(spec_padded))
#         print(f'frame_pred max = {frame_pred.max()}\tframe_pred min = {frame_pred.min()}')
        predictions = {
                'onset': frame_pred,
                'frame': frame_pred,   
                'r_adv': None
                }
        try:
            losses = {
                    'loss/train_frame': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label.reshape(-1,88)),
                    }    
        except:
            print('The prediction contains negative values')
            print(f'frame_pred min = {frame_pred.min()}')
            print(f'frame_pred max = {frame_pred.max()}')

        return predictions, losses, spec.squeeze(1)