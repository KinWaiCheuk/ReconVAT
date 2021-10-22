import pickle
import os
import numpy as np
from model import *

from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from pathlib import Path

ex = Experiment('transcription')

def transcribe2midi(data,
                    model,
                    model_type,
                    segment_samples=327680,
                    batch_size=4,
                    onset_threshold=0.5,
                    frame_threshold=0.5,
                    save_path=None,
                    reconstruction=True,
                    onset=True,
                    pseudo_onset=False,
                    rule='rule2',
                    VAT=False): 
    # Make sure the save path exists
    Path(save_path).mkdir(parents=True, exist_ok=True) # create output folder if not exist    
    model.eval()
    with torch.no_grad():       
        for i in data:
            audio = i['audio'].unsqueeze(0)
            segments = audio.unfold(1, segment_samples, segment_samples//2).squeeze(0) # faster version of enframe

            preds = {'frame':[],
                     'onset':[]}
            pointer=0
            while pointer<len(segments):
                pred = model.transcribe(segments[pointer : pointer+batch_size])
                preds['frame'].append(pred['frame'])
                preds['onset'].append(pred['onset'])
                if pointer==0: # Get the number of timesteps for pianoroll sitching later
                    T = pred['onset'].shape[1]
                pointer += batch_size

            # Calculate the start and end points for sitching
            T = pred['onset'].shape[1]
            end_pt = T//2+T//4
            start_pt = T-end_pt

            # Sitching segments back to full pianoroll
            for key in preds.keys():
                X = torch.cat(preds[key],0) # convert list into tensor
                X = X[:,:-1] # remove the last frame
                preds[key] = torch.cat((X[0,:end_pt], X[1:-1,start_pt:end_pt].flatten(0,1), X[-1,start_pt:]),0).cpu()  # faster    


            for key, value in preds.items():
                if key in ['frame','onset', 'frame2', 'onset2']:
                    value.squeeze_(0).relu_() # remove batch dim and remove make sure no negative values
                p_est, i_est = extract_notes_wo_velocity(preds['onset'], preds['frame'], onset_threshold, frame_threshold, rule=rule)


            # print(f"p_ref = {p_ref}\n p_est = {p_est}")

            t_est, f_est = notes_to_frames(p_est, i_est, preds['frame'].shape)

            scaling = HOP_LENGTH / SAMPLE_RATE

            # Converting time steps to seconds and midi number to frequency
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            t_est = t_est.astype(np.float64) * scaling
            f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

            midi_path = os.path.join(save_path, model_type+'-'+os.path.basename(i['path'])[:-4] + 'mid')
            save_midi(midi_path, p_est, i_est, [127]*len(p_est))
        
            
            
log=True
mode='imagewise'
spec='Mel'    
root = 'Application'
input_path = os.path.join(root, 'Input')
output_path = os.path.join(root, 'Output')

@ex.config
def config():
    device='cuda:0'
    model_type='ReconVAT'
    # instrument='string'

@ex.automain
def main(device, model_type):
    # Load audios from the Input files
    application_dataset = Application_Dataset(input_path,device=device)
    
    # Choose models
    if model_type=='ReconVAT':
        model = UNet((2,2),(2,2), log=log, reconstruction=True, mode=mode, spec=spec, device=device)
        weight_path = 'Weight/String_MusicNet/Unet_R_VAT-XI=1e-06-eps=1.3-String_MusicNet-lr=0.001/weight.pt'
    elif model_type=='baseline_Multi_Inst':
        model = Semantic_Segmentation(torch.empty(1,1,640,N_BINS), 1, device=device)
        weight_path = 'Weight/String_MusicNet/baseline_Multi_Inst/weight.pt'  
    
    # Load weights
    print(f'Loading model weight')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)    
    print(f'Loading done')
    print(f'Transcribing Music')
    transcribe2midi(tqdm(application_dataset), model, model_type, reconstruction=False,
                save_path=output_path)            

             