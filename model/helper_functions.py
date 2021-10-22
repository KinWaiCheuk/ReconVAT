import os
from model.dataset import *
from model.evaluate_functions import evaluate_wo_velocity
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import numpy as np

# Mac users need to uncomment these two lines
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from collections import defaultdict

def cycle(iterable):
    while True:
        for item in iterable:
            yield item

def prepare_dataset(train_on, sequence_length, validation_length, leave_one_out, refresh, device, small=False):
    train_groups, validation_groups = ['train'], ['validation'] # Parameters for MAESTRO

    if leave_one_out is not None: # It applies only to MAESTRO
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    # Choosing the dataset to use
    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
#         validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, device=device, refresh=refresh)

    elif train_on == 'MusicNet':
        dataset = MusicNet(groups=['train'], sequence_length=sequence_length, device=device, refresh=refresh)
        validation_dataset = MusicNet(groups=['test'], sequence_length=sequence_length, device=device, refresh=refresh)

    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
                           sequence_length=sequence_length, overlap=False, device=device, refresh=refresh)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'],
                                  sequence_length=validation_length, overlap=True, device=device, refresh=refresh)

    full_validation = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, device=device, refresh=refresh)
    
    return dataset, validation_dataset, full_validation

def prepare_VAT_dataset(sequence_length, validation_length, refresh, device, small=False, supersmall=False, dataset='MAPS'):
    train_groups, validation_groups = ['train'], ['validation'] # Parameters for MAESTRO

    if dataset=='MAPS':
        # Choosing the dataset to use
        if small==True:
            l_set = MAPS(groups=['AkPnBcht'],
                         sequence_length=sequence_length, overlap=False, device=device,
                         refresh=refresh, supersmall=supersmall)            
        else:
            l_set = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
                           sequence_length=sequence_length, overlap=False, device=device, refresh=refresh)    
        ul_set = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, overlap=True, device=device, refresh=refresh)        
        full_validation = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, device=device, refresh=refresh)
        
    elif dataset=='Violin':
        l_set = MusicNet(groups=['train_violin_l'],
                         sequence_length=sequence_length, device=device)            
        ul_set = MusicNet(groups=['train_violin_ul'],
                         sequence_length=sequence_length, device=device) 
#         ul_set = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
    
        validation_dataset = MusicNet(groups=['test_violin'], sequence_length=validation_length, device=device)
        full_validation = MusicNet(groups=['test_violin'], sequence_length=None, device=device)      
    elif dataset=='String':
        l_set = MusicNet(groups=['train_string_l'],
                         sequence_length=sequence_length, device=device)            
        ul_set = MusicNet(groups=['train_string_ul'],
                         sequence_length=sequence_length, device=device) 
#         ul_set = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
    
        validation_dataset = MusicNet(groups=['test_violin'], sequence_length=validation_length, device=device)
        full_validation = MusicNet(groups=['test_violin'], sequence_length=None, device=device)
        
    elif dataset=='Wind':
        l_set = MusicNet(groups=['train_wind_l'],
                         sequence_length=sequence_length, device=device)            
        ul_set = MusicNet(groups=['train_wind_ul'],
                         sequence_length=sequence_length, device=device) 
#         ul_set = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
    
        validation_dataset = MusicNet(groups=['test_wind'], sequence_length=validation_length, device=device)
        full_validation = MusicNet(groups=['test_wind'], sequence_length=None, device=device) 
        
    elif dataset=='Flute':
        l_set = MusicNet(groups=['train_flute_l'],
                         sequence_length=sequence_length, device=device)            
        ul_set = MusicNet(groups=['train_flute_ul'],
                         sequence_length=sequence_length, device=device) 
#         ul_set = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
    
        validation_dataset = MusicNet(groups=['test_flute'], sequence_length=validation_length, device=device)
        full_validation = MusicNet(groups=['test_flute'], sequence_length=None, device=device)              
        
    elif dataset=='Guqin':
        l_set = Guqin(groups=['train_l'],
                         sequence_length=sequence_length, device=device, refresh=refresh)            
        ul_set = Guqin(groups=['train_ul'],
                         sequence_length=sequence_length, device=device, refresh=refresh)            
    
        validation_dataset = Guqin(groups=['test'], sequence_length=validation_length, device=device, refresh=refresh)
        full_validation = Guqin(groups=['test'], sequence_length=None, device=device, refresh=refresh)           
    else:
        raise Exception("Please choose the correct dataset")
    
    return l_set, ul_set, validation_dataset, full_validation
     
    
def tensorboard_log(batch_visualize, model, validation_dataset, supervised_loader,
                    ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                    VAT, VAT_start, reconstruction):  
    model.eval()
    predictions, losses, mel = model.run_on_batch(batch_visualize, None, VAT)
    loss = sum(losses.values())

    if (ep)%logging_freq==0 or ep==1:
        with torch.no_grad():
            for key, values in evaluate_wo_velocity(validation_dataset, model, reconstruction=reconstruction, VAT=False).items():
                if key.startswith('metric/'):
                    _, category, name = key.split('/')
                    print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                    if ('precision' in name or 'recall' in name or 'f1' in name) and 'chroma' not in name:
                        writer.add_scalar(key, np.mean(values), global_step=ep)
#                 if key.startswith('loss/'):
#                     writer.add_scalar(key, np.mean(values), global_step=ep)                        
        model.eval()
        test_losses = eval_model(model, ep, supervised_loader, VAT_start, VAT)
        for key, values in test_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)

    if ep==1: # Showing the original transcription and spectrograms
        fig, axs = plt.subplots(2, 2, figsize=(24,8))
        axs = axs.flat
        for idx, i in enumerate(mel.cpu().detach().numpy()):
            axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
            axs[idx].axis('off')
        fig.tight_layout()

        writer.add_figure('images/Original', fig , ep)

        fig, axs = plt.subplots(2, 2, figsize=(24,4))
        axs = axs.flat
        for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
            axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
            axs[idx].axis('off')
        fig.tight_layout()
        writer.add_figure('images/Label', fig , ep)
        
        if predictions['r_adv'] is not None: 
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                x_adv = i.transpose()+predictions['r_adv'][idx].t().cpu().numpy()
                axs[idx].imshow(x_adv, vmax=1, vmin=0, cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Spec_adv', fig , ep)           

    if ep%logging_freq == 0:
        for output_key in ['frame', 'onset', 'frame2', 'onset2']:
            if output_key in predictions.keys():
                fig, axs = plt.subplots(2, 2, figsize=(24,4))
                axs = axs.flat
                for idx, i in enumerate(predictions[output_key].detach().cpu().numpy()):
                    axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure(f'images/{output_key}', fig , ep)                
        
#         fig, axs = plt.subplots(2, 2, figsize=(24,4))
#         axs = axs.flat
#         for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
#             axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
#             axs[idx].axis('off')
#         fig.tight_layout()
#         writer.add_figure('images/Transcription', fig , ep)

#         if 'onset' in predictions.keys():
#             fig, axs = plt.subplots(2, 2, figsize=(24,4))
#             axs = axs.flat
#             for idx, i in enumerate(predictions['onset'].detach().cpu().numpy()):
#                 axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
#                 axs[idx].axis('off')
#             fig.tight_layout()
#             writer.add_figure('images/onset', fig , ep)            

        if 'activation' in predictions.keys():
            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(predictions['activation'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/activation', fig , ep)   
            
        if 'reconstruction' in predictions.keys():
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(predictions['reconstruction'].cpu().detach().numpy().squeeze(1)):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Reconstruction', fig , ep)                     

        # show adversarial samples    
        if predictions['r_adv'] is not None: 
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                x_adv = i.transpose()+predictions['r_adv'][idx].t().cpu().numpy()
                axs[idx].imshow(x_adv, vmax=1, vmin=0, cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Spec_adv', fig , ep)            

        # show attention    
        if 'attention' in predictions.keys():
            fig = plt.figure(figsize=(90, 45))
            # Creating the grid for 2 attention head for the transformer
            outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
            fig.suptitle("Visualizing Attention Heads", size=20)
            attentions = predictions['attention']

            for i in range(n_heads):
                # Creating the grid for 4 samples
                inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                ax = plt.Subplot(fig, outer[i])
                ax.set_title(f'Head {i}', size=20) # This does not show up
                for idx in range(predictions['attention'].shape[0]):
                    axCenter = plt.Subplot(fig, inner[idx])
                    fig.add_subplot(axCenter)
                    attention = attentions[idx, :, i]
                    attention = flatten_attention(attention, w_size)
                    axCenter.imshow(attention.cpu().detach(), cmap='jet')


                    attended_features = mel[idx]

                    # Create another plot on top and left of the attention map                    
                    divider = make_axes_locatable(axCenter)
                    axvert = divider.append_axes('left', size='30%', pad=0.5)
                    axhoriz = divider.append_axes('top', size='20%', pad=0.25)
                    axhoriz.imshow(attended_features.t().cpu().detach(), aspect='auto', origin='lower', cmap='jet')
                    axvert.imshow(predictions['frame'][idx].cpu().detach(), aspect='auto')

                    # changing axis for the center fig
                    axCenter.set_xticks([])

                    # changing axis for the output fig (left fig)
                    axvert.set_yticks([])
                    axvert.xaxis.tick_top()
                    axvert.set_title('Transcription')

                    axhoriz.set_title(f'Attended Feature (Spec)')

                    axhoriz.margins(x=0)
                    axvert.margins(y=0)

        writer.add_figure('images/Attention', fig , ep)
        
def tensorboard_log_without_VAT(batch_visualize, model, validation_dataset, supervised_loader,
                    ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                    VAT, VAT_start, reconstruction):  
    model.eval()
    predictions, losses, mel = model.run_on_batch(batch_visualize)
    loss = sum(losses.values())

    if (ep)%logging_freq==0 or ep==1:
        with torch.no_grad():
            for key, values in evaluate_wo_velocity(validation_dataset, model, reconstruction=reconstruction, VAT=False).items():
                if key.startswith('metric/'):
                    _, category, name = key.split('/')
                    print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                    if ('precision' in name or 'recall' in name or 'f1' in name) and 'chroma' not in name:
                        writer.add_scalar(key, np.mean(values), global_step=ep)
#                 if key.startswith('loss/'):
#                     writer.add_scalar(key, np.mean(values), global_step=ep)                        
        model.eval()
        test_losses = eval_model(model, ep, supervised_loader, VAT_start, VAT)
        for key, values in test_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)

    if ep==1: # Showing the original transcription and spectrograms
        fig, axs = plt.subplots(2, 2, figsize=(24,8))
        axs = axs.flat    
        for idx, i in enumerate(mel.cpu().detach().numpy()):
            axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
            axs[idx].axis('off')
        fig.tight_layout()

        writer.add_figure('images/Original', fig , ep)

        fig, axs = plt.subplots(2, 2, figsize=(24,4))
        axs = axs.flat
        for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
            axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
            axs[idx].axis('off')
        fig.tight_layout()
        writer.add_figure('images/Label', fig , ep)
        
        if predictions['r_adv'] is not None: 
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                x_adv = i.transpose()+predictions['r_adv'][idx].t().cpu().numpy()
                axs[idx].imshow(x_adv, vmax=1, vmin=0, cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Spec_adv', fig , ep)           

    if ep%logging_freq == 0:
        for output_key in ['frame']:
            if output_key in predictions.keys():
                fig, axs = plt.subplots(2, 2, figsize=(24,4))
                axs = axs.flat
                predictions[output_key] = predictions[output_key].reshape(4,-1,88)                    
                for idx, i in enumerate(predictions[output_key].detach().cpu().numpy()):
                    axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure(f'images/{output_key}', fig , ep)                
        
#         fig, axs = plt.subplots(2, 2, figsize=(24,4))
#         axs = axs.flat
#         for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
#             axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
#             axs[idx].axis('off')
#         fig.tight_layout()
#         writer.add_figure('images/Transcription', fig , ep)

#         if 'onset' in predictions.keys():
#             fig, axs = plt.subplots(2, 2, figsize=(24,4))
#             axs = axs.flat
#             for idx, i in enumerate(predictions['onset'].detach().cpu().numpy()):
#                 axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
#                 axs[idx].axis('off')
#             fig.tight_layout()
#             writer.add_figure('images/onset', fig , ep)            

        if 'activation' in predictions.keys():
            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(predictions['activation'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/activation', fig , ep)   
            
        if 'reconstruction' in predictions.keys():
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(predictions['reconstruction'].cpu().detach().numpy().squeeze(1)):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Reconstruction', fig , ep)                     

        # show adversarial samples    
        if predictions['r_adv'] is not None: 
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                x_adv = i.transpose()+predictions['r_adv'][idx].t().cpu().numpy()
                axs[idx].imshow(x_adv, vmax=1, vmin=0, cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Spec_adv', fig , ep)            

        # show attention    
        if 'attention' in predictions.keys():
            fig = plt.figure(figsize=(90, 45))
            # Creating the grid for 2 attention head for the transformer
            outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
            fig.suptitle("Visualizing Attention Heads", size=20)
            attentions = predictions['attention']

            for i in range(n_heads):
                # Creating the grid for 4 samples
                inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                ax = plt.Subplot(fig, outer[i])
                ax.set_title(f'Head {i}', size=20) # This does not show up
                for idx in range(predictions['attention'].shape[0]):
                    axCenter = plt.Subplot(fig, inner[idx])
                    fig.add_subplot(axCenter)
                    attention = attentions[idx, :, i]
                    attention = flatten_attention(attention, w_size)
                    axCenter.imshow(attention.cpu().detach(), cmap='jet')


                    attended_features = mel[idx]

                    # Create another plot on top and left of the attention map                    
                    divider = make_axes_locatable(axCenter)
                    axvert = divider.append_axes('left', size='30%', pad=0.5)
                    axhoriz = divider.append_axes('top', size='20%', pad=0.25)
                    axhoriz.imshow(attended_features.t().cpu().detach(), aspect='auto', origin='lower', cmap='jet')
                    axvert.imshow(predictions['frame'][idx].cpu().detach(), aspect='auto')

                    # changing axis for the center fig
                    axCenter.set_xticks([])

                    # changing axis for the output fig (left fig)
                    axvert.set_yticks([])
                    axvert.xaxis.tick_top()
                    axvert.set_title('Transcription')

                    axhoriz.set_title(f'Attended Feature (Spec)')

                    axhoriz.margins(x=0)
                    axvert.margins(y=0)

        writer.add_figure('images/Attention', fig , ep)        
        
def tensorboard_log_transcriber(batch_visualize, model, validation_dataset, supervised_loader,
                    ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                    VAT, VAT_start, reconstruction):  
    model.eval()
    predictions, losses, mel = model.run_on_batch(batch_visualize, None, VAT)
    loss = sum(losses.values())

    if (ep)%logging_freq==0 or ep==1:                   
        model.eval()
        test_losses = eval_model(model, ep, supervised_loader, VAT_start, VAT)
        for key, values in test_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)

    if ep==1: # Showing the original transcription and spectrograms
        fig, axs = plt.subplots(2, 2, figsize=(24,8))
        axs = axs.flat
        for idx, i in enumerate(mel.cpu().detach().numpy()):
            axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
            axs[idx].axis('off')
        fig.tight_layout()

        writer.add_figure('images/Original', fig , ep)

        fig, axs = plt.subplots(2, 2, figsize=(24,4))
        axs = axs.flat
        for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
            axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
            axs[idx].axis('off')
        fig.tight_layout()
        writer.add_figure('images/Label', fig , ep)
                  

    if ep%logging_freq == 0:

            
        if 'reconstruction' in predictions.keys():
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(predictions['reconstruction'].cpu().detach().numpy().squeeze(1)):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Reconstruction', fig , ep)                     

        # show attention    
        if 'attention' in predictions.keys():
            fig = plt.figure(figsize=(90, 45))
            # Creating the grid for 2 attention head for the transformer
            outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
            fig.suptitle("Visualizing Attention Heads", size=20)
            attentions = predictions['attention']

            for i in range(n_heads):
                # Creating the grid for 4 samples
                inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                ax = plt.Subplot(fig, outer[i])
                ax.set_title(f'Head {i}', size=20) # This does not show up
                for idx in range(4):
                    axCenter = plt.Subplot(fig, inner[idx])
                    fig.add_subplot(axCenter)
                    attention = attentions[idx, :, i]
                    attention = flatten_attention(attention, w_size)
                    axCenter.imshow(attention.cpu().detach(), cmap='jet')


                    attended_features = mel[idx]

                    # Create another plot on top and left of the attention map                    
                    divider = make_axes_locatable(axCenter)
                    axvert = divider.append_axes('left', size='30%', pad=0.5)
                    axhoriz = divider.append_axes('top', size='20%', pad=0.25)
                    axhoriz.imshow(attended_features.t().cpu().detach(), aspect='auto', origin='lower', cmap='jet')
                    axvert.imshow(batch_visualize['frame'][idx].cpu().detach(), aspect='auto')

                    # changing axis for the center fig
                    axCenter.set_xticks([])

                    # changing axis for the output fig (left fig)
                    axvert.set_yticks([])
                    axvert.xaxis.tick_top()
                    axvert.set_title('Transcription')

                    axhoriz.set_title(f'Attended Feature (Spec)')

                    axhoriz.margins(x=0)
                    axvert.margins(y=0)

        writer.add_figure('images/Attention', fig , ep)        
                
def flatten_attention(a, w_size=31):
    w_size = (w_size-1)//2 # make it half window size
    seq_len = a.shape[0]
    n_heads = a.shape[1]
    attentions = torch.zeros(seq_len, seq_len)
    for t in range(seq_len):
        start = 0 if t-w_size<0 else t-w_size
        end = seq_len if t+w_size > seq_len else t+w_size
        if t<w_size:
            attentions[t, start:end+1] = a[t, -(end-start)-1:]
        else:
            attentions[t, start:end] = a[t, :(end-start)]
            
    return attentions        

def train_model(model, ep, loader, optimizer, scheduler, clip_gradient_norm):
    model.train()
    total_loss = 0
    batch_idx = 0
    batch_size = loader.batch_size
    total_batch = len(loader.dataset)
    # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
    for batch in loader:
        predictions, losses, _ = model.run_on_batch(batch)

        loss = sum(losses.values())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        batch_idx += 1
        print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                f'\tLoss: {loss.item():.6f}'
                , end='\r') 
    print(' '*100, end = '\r')          
    print(f'Train Epoch: {ep}\tLoss: {total_loss/len(loader):.6f}')
    return predictions, losses, optimizer

def train_VAT_model(model, iteration, ep, l_loader, ul_loader, optimizer, scheduler, clip_gradient_norm, alpha, VAT=False, VAT_start=0):
    model.train()
    batch_size = l_loader.batch_size
    total_loss = 0
    l_loader = cycle(l_loader)
    if ul_loader:
        ul_loader = cycle(ul_loader)
    for i in range(iteration):
        optimizer.zero_grad()
        
        batch_l = next(l_loader)
        
        if (ep < VAT_start) or (VAT==False):
            predictions, losses, _ = model.run_on_batch(batch_l,None, False)
        else:
            batch_ul = next(ul_loader)
            predictions, losses, _ = model.run_on_batch(batch_l,batch_ul, VAT)

#         loss = sum(losses.values())
        loss = 0
        for key in losses.keys():
            if key.startswith('loss/train_LDS'):
        #         print(key)
                loss += alpha*losses[key]/2 # No need to divide by 2 if you have only _l
            else:
                loss += losses[key]

#     loss = losses['loss/train_frame'] + alpha*(losses['loss/train_LDS_l']+losses['loss/train_LDS_ul'])/2
            
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        print(f'Train Epoch: {ep} [{i*batch_size}/{iteration*batch_size}'
                f'({100. * i / iteration:.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}\t"
#                 + f"".join([f"{k.split('/')[-1]}={v.item():.3e}\t" for k,v in losses.items()])
                , end='\r') 
    print(' '*100, end = '\r')          
    print(f'Train Epoch: {ep}\tLoss: {total_loss/iteration:.6f}')
    return predictions, losses, optimizer


def train_VAT_model_application(model, iteration, ep, l_loader, ul_loader, optimizer, scheduler, clip_gradient_norm, alpha, VAT=False, VAT_start=0):
    model.train()
    batch_size = l_loader.batch_size
    total_loss = 0
    l_loader = cycle(l_loader)
    if ul_loader:
        ul_loader = cycle(ul_loader)
    for i in range(iteration):
        optimizer.zero_grad()
        
        batch_l = next(l_loader)
        
        if (ep < VAT_start) or (VAT==False):
            predictions, losses, _ = model.run_on_batch_application(batch_l,None, False)
        else:
            batch_ul = next(ul_loader)
            predictions, losses, _ = model.run_on_batch_application(batch_l,batch_ul, VAT)

#         loss = sum(losses.values())
        loss = 0
        for key in losses.keys():
            if key.startswith('loss/train_LDS'):
        #         print(key)
                loss += alpha*losses[key]/2 # No need to divide by 2 if you have only _l
            else:
                loss += losses[key]

#     loss = losses['loss/train_frame'] + alpha*(losses['loss/train_LDS_l']+losses['loss/train_LDS_ul'])/2
            
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        print(f'Train Epoch: {ep} [{i*batch_size}/{iteration*batch_size}'
                f'({100. * i / iteration:.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}\t"
#                 + f"".join([f"{k.split('/')[-1]}={v.item():.3e}\t" for k,v in losses.items()])
                , end='\r') 
    print(' '*100, end = '\r')          
    print(f'Train Epoch: {ep}\tLoss: {total_loss/iteration:.6f}')
    return predictions, losses, optimizer



def eval_model(model, ep, loader, VAT_start=0, VAT=False):
    model.eval()
    batch_size = loader.batch_size
    metrics = defaultdict(list)
    i = 0 
    for batch in loader:
        if ep < VAT_start or VAT==False:
            predictions, losses, _ = model.run_on_batch(batch, None, False)
        else:
            predictions, losses, _ = model.run_on_batch(batch, None, True)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        print(f'Eval Epoch: {ep} [{i*batch_size}/{len(loader)*batch_size}'
                f'({100. * i / len(loader):.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}"
                , end='\r') 
        i += 1
    print(' '*100, end = '\r')          
    return metrics