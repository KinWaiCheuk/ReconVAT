import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from model import *
from evaluate import *
import pickle
import shutil
import os

ex = Experiment('evaluate')
log = True

@ex.config
def config():
    spec = 'Mel'
    attention_mode = 'onset'
    mode = 'imagewise'

    weight_file = None
    output_folder = 'results'
    inference=True
    LSTM = True
    onset = True
    device = 'cuda:0'
    refresh=False
    
    cat_feat = False
    Simple_attention=True
    
    logdir = os.path.join('results', weight_file)
    

@ex.automain
def train(spec, inference, refresh, device, logdir, weight_file, mode, LSTM, onset, Simple_attention, cat_feat): 
    
    if inference:
        inference_state = 'infer'
    else:
        inference_state = 'no_infer'
    
    print_config(ex.current_run)
    validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, device=device, refresh=refresh)

    weight_path = os.path.join('trained_weights', weight_file)

    model_type = os.path.basename(weight_path).split('-')[0]
    attention_mode = os.path.basename(weight_path).split('-')[3]
    
    if attention_mode=='feat':
        attention_mode='activation' # change the flag to match the weight name
    
    try:
        modifier = os.path.basename(weight_path).split('-')[4]
        if modifier=='no_biLSTM':
            LSTM=False
        elif modifier=='no_onset':
            onset=False
    except:
        modifier='Null'

    if model_type=='Original':
        model = OnsetsAndFrames(N_BINS, MAX_MIDI - MIN_MIDI + 1, log=log, mode=mode,
                                spec=spec, LSTM=LSTM, onset_stack=onset)    
    elif model_type=='Attention':
        print('run me')
        model = OnsetsAndFrames_with_fast_local_attn(N_BINS, MAX_MIDI - MIN_MIDI + 1,
                                                     log=log, mode=mode, spec=spec, 
                                                     LSTM=LSTM, onset_stack=onset,
                                                     attention_mode=attention_mode)         
    elif model_type=='Simple':
        model = SimpleModel(N_BINS, MAX_MIDI - MIN_MIDI + 1, log=log, mode=mode, spec=spec,
                            device=device, w_size=int(modifier[2:]), attention=Simple_attention, layers=1,
                            cat_feat=False, onset=False)
    model.to(device)
    model.load_my_state_dict(torch.load(weight_path+'.pt'))

    
    
    with torch.no_grad():
        model.eval()
        metrics = evaluate_wo_velocity(tqdm(validation_dataset), model, reconstruction=False,
                                       save_path=os.path.join(logdir,f'./MIDI_results-{inference_state}-{modifier}'),
                                       onset=inference)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values)*100:.3f} ± {np.std(values)*100:.3f}')
    export_path = os.path.join(logdir, f'result_dict_{inference_state}-{modifier}')   
    pickle.dump(metrics, open(export_path, 'wb'))