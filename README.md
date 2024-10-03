# Unsupervised One-Shot Style Adaptation for Visual-Optical and Thermal Infrared Object Detection

This repository contains the official implementation of our paper [OSSA: Unsupervised One-Shot Style Adaptation](https://arxiv.org/abs/2410.00900). 

![combined_video](https://github.com/RobinGerster7/OSSA/assets/164496870/ee853980-e7a4-48ff-9800-126bd8d75913)

*The video shows the detections of a FRCNN with a ResNet-50 backbone trained on the Sim10k dataset and tested on the Cityscapes dataset. The baseline version is displayed on the left, while OSSA is shown on the right. We see that OSSA is more effective at detecting cars.*



## Installation

To begin, we need to install MMDetection. Follow the steps below to set up the environment:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install cityscapesScripts

pip install -U openmim
mim install mmengine

pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

Move `resnet_ossa.py` from `custom/` to `mmdetection/mmdet/models/backbones`. Then, update the `__init__.py` in the same directory to include `from .resnet_ossa import ResNetOSSA` and add `"ResNetOSSA"` to `__all__`.



## Dataset Preparation

### Cityscapes Dataset

#### Download and Extraction

1. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from the [Cityscapes Dataset Downloads Page](https://www.cityscapes-dataset.com/downloads/).
2. Extract both archives into `datasets/cityscapes`.

#### Generating COCO Style Annotations

Within the OSSA directory, execute:

```bash
python ./tools/cityscapes2coco.py ../datasets/cityscapes -o ../datasets/cityscapes/annotations
```

**Note:** Post-annotation generation, the `datasets/cityscapes/gtFine` directory can be safely removed.

### Foggy Cityscapes Dataset

#### Download and Setup

1. Return to the [Cityscapes Downloads Page](https://www.cityscapes-dataset.com/downloads/) and download `leftImg8bit_trainvaltest_foggy.zip`.
2. Ensure `gtFine_trainvaltest.zip` is still present and extract both zips into `datasets/foggy_cityscapes`.

#### Processing

Execute the following to refine the dataset:

```bash
python ./tools/prepare_foggy_cityscapes.py
```

Then, to generate COCO-style annotations:

```bash
python ./tools/cityscapes2coco.py ../datasets/foggy_cityscapes -o ../datasets/foggy_cityscapes/annotations --img-dir leftImg8bit_foggy
``` 

#### Finalizing

Rename the `leftImg8bit_foggy` directory to `leftImg8bit` within `datasets/foggy_cityscapes`.

---

### Sim10k Dataset

#### Download and Setup

1. Visit [Sim10k Downloads Page](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and install the 10k images and annotations.
2. Extract both zip files to the datasets/ folder.

#### Processing

Exectute the following to generate the coco style annotations:

```bash
python ./tools/sim2coco.py --ann_dir ../datasets/VOC2012/Annotations --output ../datasets/VOC2012/annotations.coco.json
```

#### Finalizing
Rename VOC2012 to sim10k and delete the VOC2012/Annotations directory.

## Training OSSA
For sim10k->cityscapes adaptation use:
```bash
python ./mmdetection/tools/train.py ./configs/prototype_constructors/frcnn_ossa_proto_city.py
python ./mmdetection/tools/train.py ./configs/experiments/frcnn_ossa_sim2city.py
```

For cityscapes->foggy adaptation use:
```bash
python ./mmdetection/tools/train.py ./configs/prototype_constructors/frcnn_ossa_proto_foggy.py
python ./mmdetection/tools/train.py ./configs/experiments/frcnn_ossa_city2foggy.py
```

### Testing OSSA
Below is an example of how to test OSSA given weights.
```bash
python mmdetection/tools/test.py configs/experiments/frcnn_ossa_city2foggy.py city2foggy.pth
```

## Model Weights

Below is the table with links to download the trained models and their performance metrics:

| Dataset Adaptation             | Model Weights                                                                                     | mAP50 |
|--------------------------------|---------------------------------------------------------------------------------------------------|-------|
| Sim10k -> Cityscapes           | [Google Drive Link](https://drive.google.com/file/d/1H_2v7j-Q7fZBrsjXk_8P44JYuflNuikg/view?usp=sharing)         | 53.1  |
| Cityscapes -> Foggy Cityscapes | [Google Drive Link](https://drive.google.com/file/d/1UsrPd6wC9eltL4PJLnP3rLH0mw9X7LNM/view?usp=sharing)         | 40.3  |
| M3FD Visual -> Thermal         | [Google Drive Link](https://drive.google.com/file/d/1HYqW_L5PMN-42FTk1baClHmx8DsPxH0A/view?usp=sharing)               | 35.2  |

