# WaterDiff: Physical Prior-based Latent Diffusion Model for Underwater Image Enhancement

See [environment.yaml](./environment.yaml) for requirements on packages. Simple installation:

```
conda env create -f environment.yaml
```

### Test sets:


To make use of the [evaluate.py](evaluate.py)  the dataset folder names should be lower-case and structured as follows.

```
└──── <data directory>/
    ├──── UIEB_R90/
    |   ├──── input_256/
    |   |   ├──── 01.png/
    |   |   ├──── ...
    |   |   └──── 90.png/
    |   ├──── target_256/
    |   |   ├──── 01.png/
    |   |   ├──── ...
    |   |   └──── 90.png/
    |   ├──── transmap_256/
		    ├──── 01.png/
    	    ├──── ...
		    └──── 90.png/
   
```

### Transmap

```
Please use IR_GDCP.m to generate your transmission map.

Please cite the related paper if you use this code to generate your transmission map. Thanks. 
"Generalization of the Dark Channel Prior for Single Image Restoration". 
```

## Evaluation

To resume from a checkpoint file, simply use the `--resume` argument in [evaluate.py](evaluate.py) to specify the checkpoint.

## Training

LDMVFI is trained in two stages, where the VQ-FIGAN and the denoising U-Net are trained separately.

### VQ-FIGAN

```
python main.py --base configs/autoencoder/vq_f16×16×3.yaml -t --gpus 0,
```

### Denoising U-Net

```
python main.py --base configs/ldm/vquie-f16-c256-cross.yaml -t --gpus 0,
```

These will create a `logs/` folder within which the corresonding directories are created for each experiment. The log files from training include checkpoints, images and tensorboard loggings.

To resume from a checkpoint file, simply use the `--resume` argument in [main.py](main.py) to specify the checkpoint.

## Acknowledgement

Our code is adapted from the original [latent-diffusion](https://github.com/danier97/LDMVFI) repository. We thank the authors for sharing their code.
