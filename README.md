# StyleGAN2 for practice

<p align='center'><img src='_out/mix_urart6-1024-f-4096-1024.jpg' /></p>

This version of famous [StyleGAN2] is intended mostly for fellow artists and students, who rarely look at scientific metrics, but rather need a working tool. At least, this is what I use daily myself. For more explicit details refer to the original implementations. 

## Features
* inference (image generation) in arbitrary resolution (may cause artifacts!)
* non-square aspect ratio support (picked from dataset)
  (resolution must be divisible by 2**n, such as 512x256, 1280x768, etc.)
* transparency (alpha channel) support (picked from dataset)
* freezing lower D layers for better finetuning (from [Freeze the Discriminator])

Windows batch-commands for main tasks, such as ::
* video rendering with slerp/cubic/gauss trajectory smoothing (requires [FFMPEG])
* animated "playback" of saved latent snapshots, etc.

also, from [Data-Efficient GANs] ::
* differential augmentation for fast training on small datasets (~100 images)
* support of custom CUDA-compiled TF ops and (slower) Python-based reference ops

also, from [Peter Baylies] and [skyflynil] ::
* non-progressive configs (E,F) with single-scale datasets
* raw JPG support in TFRecords dataset (dramatic savings in disk space & dataset creation time)
* conditional support 
* vertical mirroring augmentation

## Usage examples

* Put your files in `data` as subfolder. If needed, crop square fragments from `source` video or directory with images (feasible method, if you work with patterns or shapes, rather than compostions):
```
 multicrop.bat source 512 256 
```
This will cut every source image (or video frame) into 512x512px fragments, overlapped with shift 256px by X and Y. Result will be in directory `source-sub`, rename it as you wish. Non-square dataset should be prepared separately.

* Make compact TFRecords dataset from directory with JPG images `data/mydata`:
```
 prepare_dataset.bat mydata 
```
This will create file `mydata-512x512.tfr` in `data` directory (if your dataset resolution is 512x512). For images with alpha channel remove `--jpg` option from this bat-file, and also `--jpg_data` option from `train.bat` or `train_resume.bat` files. 

* Train StyleGAN2 on prepared dataset:
```
 train.bat mydata 
```
This will run training process, according to the options in `src/train.py`. If there's no TFRecords file from previous step, it will be created here. The training results (models and samples) are saved under the `train` directory, similar to original Nvidia approach. Only newest configs E and F are used in this repo (default is F; set `--config E` if you face OOM issue). 

Please note: we save both full models (containing G/D/Gs networks for further training) as `snapshot-baseresolution-config-kimg.pkl`, and compact models (containing only Gs network for inference) as  `dataset-baseresolution-config-kimg.pkl`, e.g. `mydata-512-f-0360.pkl`. For non-square dataset, the name will be extended to `dataset-baseresolution-config-initialXY-kimg.pkl` (e.g.`mydata-512-f-3x4-0360.pkl`). Changing this naming may break other scripts behaviour! 

For small datasets (100x images instead of 10000x) one may add `--d_aug` option to use [Differential Augmentation] for more effective training. 
The length of the training is defined by `--lod_step_kimg XX` option. It's kind of legacy from [progressive GAN] and defines one step of progressive training. The network with base resolution 1024px will be trained for 20 such steps, for 512px - 18 steps, et cetera. Reasonable value for big datasets is 300-600, while in `--d_aug` mode 20-40 is sufficient.

* Resume training on `mydata` dataset from the last saved model at `train/000-mydata-512-f` directory:
```
 train_resume.bat mydata 000-mydata-512-f
```

* Uptrain (finetune) trained model on new data:
```
 train_resume.bat newdata 000-mydata-512-f --finetune 
```
`--finetune` option only sets fixed learning rate and some fake high-kimg steps (it's also legacy from ProGAN/StyleGAN). There's no specific schedule in this case, you may stop when you're ok with the results (it's better to set low `lod_step_kimg` to follow the process). There's also `--freezeD` option, supposedly enhancing finetuning (not tested).

* Reduce full model (containing G/D/Gs networks) to a compact one (Gs only) for inference:
```
 reduce_model.bat snapshot-512-f-xxx.pkl 
```
The result is saved in `models` directory. Useful for foreign downloaded models.

* Generate smooth animation between random latent points:
```
 gen.bat ffhq-1024-f 1280-720 500-20 
```
This will load `ffhq-1024-f.pkl` from `model` directory and make a looped 1280x720 px video of 500 frames, with interpolation step of 20 frames between keypoints. Please note: omitting `.pkl` extension enables custom resolution. Using full filename with extention will load the network from PKL itself (useful to test foreign downloaded models). There are `--cubic` and `--gauss` options for animation smoothing, and few `--scale_type` choices. Besides video/sequence output, this command will also save all traversed dlatent points as Numpy array in `*.npy` file.

* Project external images onto StyleGAN2 model dlatent space (ensure first that `vgg16_zhang_perceptual.pkl` is downloaded from Git LFS to `models/vgg`):
```
 project.bat yourmodel.pkl imagedir 
```
The result (found dlatent points as Numpy arrays in `*.npy` files, and video/still previews) will be saved to `_out/proj` directory. 

* Generate smooth animation between saved dlatent points:
```
 play_dlatents.bat ffhq-1024-f mynpy 50 1920-1080 
```
This will load saved dlatent points from `_in/mynpy` and produce a smooth looped animation between them (with resolution 1920x1080 and interpolation step of 50 frames). `mynpy` may be a file or a directory with `*.npy` files. To select only few frames from a sequence `somename.npy`, create text file with comma-delimited frame numbers and save it as `somename.txt` in the same directory (check examples for FFHQ model). You can also "style" the result: setting `--style_npy_file blonde458.npy` will load dlatent from `blonde458.npy` and apply it to higher layers, producing some visual similarity. `--cubic` smoothing is also applicable here. 

* Generate animation from saved dlatent point and feature directions (say, aging/smiling/etc for faces model):
```
 play_vectors.bat ffhq-1024-f.pkl blonde458.npy vectors_ffhq 
```
This will load base dlatent point from `_in/blonde458.npy` and move it along latent direction vectors from `_in/vectors_ffhq`, one by one. Result is saved as a looped video. 

[StyleGAN2]: <https://github.com/NVlabs/stylegan2>
[Peter Baylies]: <https://github.com/pbaylies/stylegan2>
[skyflynil]: <https://github.com/skyflynil/stylegan2>
[Data-Efficient GANs]: <https://github.com/mit-han-lab/data-efficient-gans>
[Differential Augmentation]: <https://github.com/mit-han-lab/data-efficient-gans>
[Freeze the Discriminator]: <https://arxiv.org/abs/2002.10964>
[FFMPEG]: <https://ffmpeg.org/download.html>
[progressive GAN]: <https://github.com/tkarras/progressive_growing_of_gans>
