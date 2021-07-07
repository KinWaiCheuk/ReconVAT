# ReconVAT
This repository is for the paper **ReconVAT: A Semi-Supervised Automatic Music Transcription Framework Towards Real-World Applications**.

Demo page is available at: https://kinwaicheuk.github.io/ReconVAT/

Supplementary Materials: TODO


## Requirement
You can install the following libraries at once via `pip install -r requirements.txt`.


## Using pretrained models

For your convenience, we have provided 3 example audio clips for you to try our models out. But to transcribe your own music, you need to first downsample them to 16kHz and save them as Flac format. Then simply put your audio clips in the path `Application/Input`, then run the follow code:
```python
python transcribe_files.py with model_type=<arg> device=<arg>
```

* `model_type`: Pick the model to transcribe your music. `ReconVAT` or `baseline_Multi_Inst`. Default is `ReconVAT`.
* `device`: the device to be trained on. Either `cpu` or `cuda:0`. Default is `cuda:0`

You might also need to install ffmpeg in order to do audio downsampling.
On macOS:
```python
brew install ffmpeg
```

On Linux:
```
Apt-get install ffmpeg
```

## Training from scratch
### Step1: Downloading Dataset 
MAPS dataset (as labelled dataset in our experiments): [download](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/)

MAESTRO (we use v2.0.0 as our unlabelled dataset in our experiments): [download](https://magenta.tensorflow.org/datasets/maestro)

MusicNet dataset (for training strings and woodwinds): [download](https://homes.cs.washington.edu/~thickstn/musicnet.html)

After downloading these dataset, unzip them to their respective folders `MAPS`, `MAESTRO`, and `MusicNet`.

### Step2: Preprocessing
Our model takes 16kHz audio as the input, therefore we need to downsample all the audio clips first. Our model also takes tsv files as the labels, so we also need to convert midi files into tsv files.

These preprocessing functions can be found in the jupyter notebook named as `Preprocessing.ipynb`.

When the dataset is ready, the PyTorch Dataset class should be able toload these datasets without errors.

### Step3: Training the model
The python script can be run using using the sacred syntax `with`.

Unet_VAT mode:
```python
python train_UNet_VAT.py with train_on=<arg> small=<arg> VAT=<arg> reconstruction=<arg> device=<arg>
```

Unet_VAT with the onset module:
```python
python train_UNet_Onset_VAT.py with train_on=<arg> small=<arg> VAT=<arg> reconstruction=<arg> device=<arg>
```

Baseline model Multi-instrument:
```python
python train_baseline_Multi_Inst.py with train_on=<arg> small=<arg> device=<arg>
```

Onsets and Frames: (VAT can be activated in this baseline model, but according to our experiments, VAT does not work with this baseline model)
```python
python train_baseline_onset_frame_VAT.py with train_on=<arg> small=<arg> device=<arg>
```

The following two baseline model requires a huge amount of GPU memory

Thickstun:
```python
python train_baseline_Thickstun.py with train_on=<arg> small=<arg> device=<arg>
```

Prestack:
```python
python train_baseline_Prestack.py with train_on=<arg> small=<arg> device=<arg>
```


* `train_on`: the dataset to be trained on. Either `MAPS` or `String` or `Wind`
* `small`: Activate the small version of MAPS. `True` or `False`
* `supersmall`: Activate the oneshot version of MAPS. The `small` argument has to be `True` in order for this argument to be useful
* `reconstruction`: to include the reconstruction loss or not. Either `True` or `False`
* `VAT`: VAT module, `True` or `False`
* `device`: the device to be trained on. Either `cpu` or `cuda:0`

## Evaluating the model and exporting the midi files

```python
python evaluate.py with weight_file=<arg> reconstruction=<arg> device=<arg>
```

* `weight_file`: The weight files should be located inside the `trained_weight` folder
* `dataset`: which dataset to evaluate on, can be either `MAPS` or `MAESTRO` or `MusicNet`.
* `device`: the device to be trained on. Either `cpu` or `cuda:0`

The transcripted midi files, accuracy reports are saved inside the `results` folder.