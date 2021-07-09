import os

from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model import *
ex = Experiment('train_original')

# parameters for the network
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 1
output_channel = 2
logging_freq = 10
saving_freq = 10

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
    spec = 'Mel'
    resume_iteration = None
    train_on = 'String'
    n_heads=4
    position=True
    iteration = 10
    VAT_start = 0
    alpha = 1
    VAT=True
    XI= 1e-6
    eps=1.3
    small = True
    KL_Div = False
    reconstruction = False

    
    batch_size = 1
    train_batch_size = 1
    sequence_length = 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 20000        
    step_size_up = 100    
    max_lr = 1e-4 
    learning_rate = 0.0001  
#     base_lr = learning_rate

    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    refresh = False

    logdir = f'{root}/baseline_ThickStun-lr={learning_rate}'+ datetime.now().strftime('%y%m%d-%H%M%S')
        
    ex.observers.append(FileStorageObserver.create(logdir)) # saving source code
        
@ex.automain
def train(spec, resume_iteration, train_on, batch_size, sequence_length,w_size, n_heads, small, train_batch_size,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out, position, alpha, KL_Div,
          clip_gradient_norm, validation_length, refresh, device, epoches, logdir, log, iteration, VAT_start, VAT, XI, eps,
          reconstruction): 
    print_config(ex.current_run)

    
    supervised_set, unsupervised_set, validation_dataset, full_validation = prepare_VAT_dataset(
                                                                          sequence_length=sequence_length,
                                                                          validation_length=sequence_length,
                                                                          refresh=refresh,
                                                                          device=device,
                                                                          small=small,
                                                                          supersmall=True,
                                                                          dataset=train_on)  
    


    if len(validation_dataset)>4:
        val_batch_size=4
    else:
        val_batch_size = len(validation_dataset)
    supervised_loader = DataLoader(supervised_set, train_batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, val_batch_size, shuffle=False, drop_last=True)
    batch_visualize = next(iter(valloader)) # Getting one fixed batch for visualization    
    
    ds_ksize, ds_stride = (2,2),(2,2)     
    model = Thickstun()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    
    summary(model)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,cycle_momentum=False)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    
    print(f'supervised_loader')
    
    for ep in range(1, epoches+1):
        predictions, losses, optimizer = train_model(model, ep, supervised_loader,
                                                             optimizer, scheduler, clip_gradient_norm)            
        loss = sum(losses.values())
        
        # Logging results to tensorboard
        if ep == 1:
            writer = SummaryWriter(logdir) # create tensorboard logger     
        tensorboard_log_without_VAT(batch_visualize, model, validation_dataset, supervised_loader,
                        ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                        False, VAT_start, reconstruction)
          
        
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


