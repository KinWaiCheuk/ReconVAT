import os

from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model import *
ex = Experiment('train_original')

# parameters for the network
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 1
output_channel = 2
logging_freq = 100
saving_freq = 1000


@ex.config
def config():
    root = 'runs'
    # logdir = f'runs_AE/test' + '-' + datetime.now().strftime('%y%m%d-%H%M%S')
    # Choosing GPU to use
#     GPU = '0'
#     os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    onset_stack=True
    device = 'cuda:0'
    log = True
    w_size = 31
    model_complexity = 48
    spec = 'Mel'
    resume_iteration = None
    train_on = 'String'
    iteration = 10
    alpha = 1
    VAT=False
    XI= 1e-6
    eps=1e-1
    VAT_mode = 'all'
    model_name = 'onset_frame'
    VAT_start = 0
    small = True

    
    batch_size = 8
    train_batch_size = 8
    sequence_length = 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 20000        
    learning_rate = 5e-4
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    refresh = False

    logdir = f'{root}/baseline_Onset_Frame-'+ datetime.now().strftime('%y%m%d-%H%M%S')
        
    ex.observers.append(FileStorageObserver.create(logdir)) # saving source code
        
@ex.automain
def train(spec, resume_iteration, train_on, batch_size, sequence_length,w_size, model_complexity, VAT_mode, VAT_start,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out, alpha, model_name, train_batch_size,
          clip_gradient_norm, validation_length, refresh, device, epoches, logdir, log, iteration, VAT, XI, eps, small): 
    print_config(ex.current_run)
    supervised_set, unsupervised_set, validation_dataset, full_validation = prepare_VAT_dataset(sequence_length=sequence_length,
                                                                              validation_length=sequence_length,
                                                                              refresh=refresh,
                                                                              device=device,
                                                                              small=small,
                                                                              supersmall=False,
                                                                              dataset=train_on)
    
    MAPS_supervised_set, MAPS_unsupervised_set, MAPS_validation_dataset, _ = prepare_VAT_dataset(
                                                                          sequence_length=sequence_length,
                                                                          validation_length=sequence_length,
                                                                          refresh=refresh,
                                                                          device=device,
                                                                          small=small,
                                                                          supersmall=True,
                                                                          dataset='MAPS')
    
    supervised_set = ConcatDataset([supervised_set, MAPS_supervised_set])
    unsupervised_set = ConcatDataset([unsupervised_set, MAPS_unsupervised_set])    
    
    unsupervised_loader = DataLoader(unsupervised_set, batch_size, shuffle=True, drop_last=True)
    supervised_loader = DataLoader(supervised_set, train_batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, len(validation_dataset), shuffle=False, drop_last=True)
    batch_visualize = next(iter(valloader)) # Getting one fixed batch for visualization    
    
    if resume_iteration is None:
        
        if model_name=='onset_frame':
            model = OnsetsAndFrames_VAT_full(N_BINS, MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity,
                                              log=log, mode=mode, spec=spec, XI=XI, eps=eps, VAT_mode=VAT_mode)
        elif model_name=='frame':
            model = Frame_stack_VAT(N_BINS, MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity,
                                    log=log, mode=mode, spec=spec, XI=XI, eps=eps, VAT_mode=VAT_mode)
        elif model_name=='onset':
            model = Onset_stack_VAT(N_BINS, MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity,
                                    log=log, mode=mode, spec=spec, XI=XI, eps=eps, VAT_mode=VAT_mode)  
        elif model_name=='attention':
            model = Frame_stack_attention_VAT(N_BINS, MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity,
                                    log=log, mode=mode, spec=spec, XI=XI, eps=eps, VAT_mode=VAT_mode)                   
            
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else: # Loading checkpoints and continue training
        trained_dir='trained_MAPS' # Assume that the checkpoint is in this folder
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))

    for ep in range(1, epoches+1):
        model.train()
        predictions, losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, unsupervised_loader,
                                                     optimizer, scheduler, clip_gradient_norm, alpha, VAT, VAT_start)
        loss = sum(losses.values())      
        
        # Logging results to tensorboard
        if ep == 1:
            writer = SummaryWriter(logdir) # create tensorboard logger     
        if ep < VAT_start or VAT==False:
            tensorboard_log(batch_visualize, model, validation_dataset, supervised_loader,
                            ep, logging_freq, saving_freq, 8, logdir, w_size, writer, False, VAT_start, reconstruction=False)
        else:
            
            tensorboard_log(batch_visualize, model, validation_dataset, supervised_loader,
                            ep, logging_freq, saving_freq, 8, logdir, w_size, writer, True, VAT_start, reconstruction=False)            
        
        # Saving model
        if (ep)%saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        for key, value in {**losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)           
                
    # Evaluating model performance on the full MAPS songs in the test split     
    print('Training finished, now evaluating on the MAPS test split (full songs)')
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(full_validation), model, reconstruction=False,
                                       save_path=os.path.join(logdir,'./MIDI_results'))
        
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
         
    export_path = os.path.join(logdir, 'result_dict')    
    pickle.dump(metrics, open(export_path, 'wb'))




