import json
import os
from abc import abstractmethod
from glob import glob
import sys
import pickle
import pandas as pd


import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []
        print(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in self.groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
#             print(f'step_begin = {step_begin}')
            
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
#             print(f'begin = {begin}')
            end = begin + self.sequence_length
    
            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
            result['start_idx'] = begin
            
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        
#         print(f'audio size = {audio.shape}')
#         print(f'label size = {label.shape}')
        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='../../public_data/MAESTRO/', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v2.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='./MAPS', groups=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', supersmall=False):
        self.overlap = overlap
        self.supersmall = supersmall
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        if self.overlap==False:
            with open('overlapping.pkl', 'rb') as f:
                test_names = pickle.load(f)
            filtered_flacs = []    
            for i in flacs:
                if any([substring in i for substring in test_names]):
                    pass
                else:
                    filtered_flacs.append(i)
            flacs = sorted(filtered_flacs)
            if self.supersmall==True:
#                 print(sorted(filtered_flacs))
                flacs = [sorted(filtered_flacs)[3]]
        # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
#         print(flacs)
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))

class MusicNet(PianoRollAudioDataset):
    def __init__(self, path='./MusicNet', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'test']

    def read_id(self, path, group, mode):
        train_meta = pd.read_csv(os.path.join(path,f'{mode}_metadata.csv'))
        return train_meta[train_meta['ensemble'].str.contains(group)]['id'].values    
    
    def appending_flac_tsv(self, id_list, mode):
        flacs = []
        tsvs = []
        for i in id_list:
            flacs.extend(glob(os.path.join(self.path, f"{mode}_data", f"{i}.flac")))
            tsvs.extend(glob(os.path.join(self.path, f"tsv_{mode}_labels/{i}.tsv")))
        flacs = sorted(flacs)
        tsvs = sorted(tsvs)   
        return flacs, tsvs

    def files(self, group):
        string_keys = ['Solo Violin', 'Violin and Harpsichord',
                'Accompanied Violin', 'String Quartet',
                'String Sextet', 'Viola Quintet',
                'Solo Cello', 'Accompanied Cello']      
        
        wind_keys = ['Accompanied Clarinet', 'Clarinet Quintet',
                    'Pairs Clarinet-Horn-Bassoon', 'Clarinet-Cello-Piano Trio',
                    'Wind Octet', 'Wind Quintet']
        
        
        
        train_meta = pd.read_csv(os.path.join(self.path,f'train_metadata.csv'))
        if group == 'small test':
            types = ('2303.flac', '2382.flac', '1819.flac')
            flacs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'test_data', i)))
            flacs = sorted(flacs)
            tsvs = sorted(glob(os.path.join(self.path, f'tsv_test_labels/*.tsv')))   
        elif group == 'train_string_l':
            types = np.array([0])
            for key in string_keys:
                l= train_meta[train_meta['ensemble'].str.contains(key)]['id'].values[:1]
                types = np.concatenate((types,l))
            types = np.delete(types, 0)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')
        elif group == 'train_string_ul':
            types = np.array([0])
            for key in string_keys:
                l= train_meta[train_meta['ensemble'].str.contains(key)]['id'].values[1:]
                types = np.concatenate((types,l))
            types = np.delete(types, 0)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')                
            
        elif group == 'train_violin_l':
            type1 = self.read_id(self.path, 'Solo Violin', 'train')
            type2 = self.read_id(self.path, 'Accompanied Violin', 'train')    
            types = np.concatenate((type1,type2))            
            flacs, tsvs = self.appending_flac_tsv(types, 'train')
            
        elif group == 'train_violin_ul':
            type1 = self.read_id(self.path, 'String Quartet', 'train')
            type2 = self.read_id(self.path, 'String Sextet', 'train')
            types = np.concatenate((type1,type2))
            flacs, tsvs = self.appending_flac_tsv(types, 'train')  
            
        elif group == 'test_violin':
            types = ('2106', '2191', '2298', '2628')
            flacs, tsvs = self.appending_flac_tsv(types, 'test')
                      
        elif group == 'train_wind_l':
            types = np.array([0])
            for key in wind_keys:
                l= train_meta[train_meta['ensemble'].str.contains(key)]['id'].values[:1]
                types = np.concatenate((types,l))
            types = np.delete(types, 0)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')
        elif group == 'train_wind_ul':
            types = np.array([0])
            for key in wind_keys:
                l= train_meta[train_meta['ensemble'].str.contains(key)]['id'].values[1:]
                types = np.concatenate((types,l))
            types = np.delete(types, 0)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')                   
            
        elif group == 'test_wind':
            types = ('1819', '2416')
            flacs, tsvs = self.appending_flac_tsv(types, 'test') 
            
        elif group == 'train_flute_l':
            types = ('2203',)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')     
        elif group == 'train_flute_ul':
            types = np.array([0])
            for key in wind_keys:
                l= train_meta[train_meta['ensemble'].str.contains(key)]['id'].values[:]
                types = np.concatenate((types,l)) 
            types = np.delete(types, 0)
            types = np.concatenate((types,('2203',)))
            
            flacs, tsvs = self.appending_flac_tsv(types, 'train')
            
        elif group == 'test_flute':
            types = ('2204',)
            flacs, tsvs = self.appending_flac_tsv(types, 'train')             
            
        else:
            types = self.read_id(self.path, group, 'train')
            flacs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'train_data', f"{i}.flac")))
            flacs = sorted(flacs)
            tsvs = sorted(glob(os.path.join(self.path, f'tsv_train_labels/*.tsv')))   
#         else:
#             flacs = sorted(glob(os.path.join(self.path, f'{group}_data/*.flac')))
#             tsvs = sorted(glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))            
#         else:
#             flacs = sorted(glob(os.path.join(self.path, f'{group}_data/*.flac')))
#             tsvs = sorted(glob(os.path.join(self.path, f'tsv_{group}_labels/*.tsv')))

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return zip(flacs, tsvs)
    
    
class Guqin(PianoRollAudioDataset):
    def __init__(self, path='./Guqin', groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train_l', 'train_ul', 'test']

    def read_id(self, path, group, mode):
        train_meta = pd.read_csv(os.path.join(path,f'{mode}_metadata.csv'))
        return train_meta[train_meta['ensemble'].str.contains(group)]['id'].values    
    
    def appending_flac_tsv(self, id_list, mode):
        flacs = []
        tsvs = []
        for i in id_list:
            flacs.extend(glob(os.path.join(self.path, f"{mode}_data", f"{i}.flac")))
            tsvs.extend(glob(os.path.join(self.path, f"tsv_{mode}_labels/{i}.tsv")))
        flacs = sorted(flacs)
        tsvs = sorted(tsvs)   
        return flacs, tsvs

    def files(self, group):
        if group=='train_l':
            types = ['jiou', 'siang', 'ciou', 'yi', 'yu', 'feng', 'yang'] 
            flacs = []
            tsvs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'audio', i + '.flac')))
                tsvs.extend(glob(os.path.join(self.path, 'tsv_label', i + '.tsv')))
            flacs = sorted(flacs)
            tsvs = sorted(tsvs)          
            return zip(flacs, tsvs)
            
        elif group == 'train_ul':
            types = [
                    ] 
            flacs = []
            tsvs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'audio', i + '.flac')))
                tsvs.extend(glob(os.path.join(self.path, 'tsv_label', i + '.tsv')))
            flacs = sorted(flacs)
            tsvs = sorted(tsvs)       
            return zip(flacs, tsvs)
            
        elif group == 'test':
            types = ['gu', 'guan', 'liang',
                     ]
            flacs = []
            tsvs = []
            for i in types:
                flacs.extend(glob(os.path.join(self.path, 'audio', i + '.flac')))
                tsvs.extend(glob(os.path.join(self.path, 'tsv_label', i + '.tsv')))
            flacs = sorted(flacs)
            tsvs = sorted(tsvs) 
            print(f'flacs = {flacs}')
            print(f'tsvs = {tsvs}')
            return zip(flacs, tsvs)
            
            
        else:
            raise Exception("Please choose a valid group")


class Corelli(PianoRollAudioDataset):
    def __init__(self, path='./Application_String', groups=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', supersmall=False):
        self.overlap = overlap
        self.supersmall = supersmall
        super().__init__(path, groups, sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['op6_no1', 'op6_no2', 'op6_no3']

    def files(self, group):
        flacs = glob(os.path.join(self.path, group, '*.flac'))
        
        if self.overlap==False:
            with open('overlapping.pkl', 'rb') as f:
                test_names = pickle.load(f)
            filtered_flacs = []    
            for i in flacs:
                if any([substring in i for substring in test_names]):
                    pass
                else:
                    filtered_flacs.append(i)
            flacs = sorted(filtered_flacs)
            if self.supersmall==True:
#                 print(sorted(filtered_flacs))
                flacs = [sorted(filtered_flacs)[3]]
        # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
#         print(flacs)
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))
        
        return sorted(zip(flacs, tsvs))
    
    
class Application_Dataset(Dataset):
    def __init__(self, path, seed=42, device='cpu'):
        self.path = path
        
        self.device = device

        self.data = []
        for input_files in tqdm(self.files(path), desc='Loading files'): #self.files is defined in MAPS class
            self.data.append(self.load(input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        audio_length = len(data['audio'])
        result['audio'] = data['audio'].to(self.device)

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        return result

    def __len__(self):
        return len(self.data)


    @abstractmethod
    def files(self, group):
        # Only need to load flac files
        flacs = glob(os.path.join(self.path, '*.flac'))
        flacs.extend(glob(os.path.join(self.path, '*.wav'))) # If there are wav files, also load them
        assert(all(os.path.isfile(flac) for flac in flacs))
        
        return flacs

    def load(self, audio_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        # 
        assert sr == SAMPLE_RATE, f'Please make sure the sampling rate is 16k.\n{saved_data_path} has a sampling of {sr}'
 
            
        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)

        data = dict(path=audio_path, audio=audio)
        return data
    
    

class Application_Wind(PianoRollAudioDataset):
    def __init__(self, path='./Application_Wind', groups=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', supersmall=False):
        self.overlap = overlap
        self.supersmall = supersmall
        super().__init__(path, groups, sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['dummy']

    def files(self, group):
        flacs = glob(os.path.join(self.path, '*.flac'))
        
        if self.overlap==False:
            with open('overlapping.pkl', 'rb') as f:
                test_names = pickle.load(f)
            filtered_flacs = []    
            for i in flacs:
                if any([substring in i for substring in test_names]):
                    pass
                else:
                    filtered_flacs.append(i)
            flacs = sorted(filtered_flacs)
            if self.supersmall==True:
#                 print(sorted(filtered_flacs))
                flacs = [sorted(filtered_flacs)[3]]
        # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
#         print(flacs)
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))
        
        return sorted(zip(flacs, tsvs))