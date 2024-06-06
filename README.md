# Combined Video

This repository contains the official implementation of our paper [link to be provided]. 

![Combined Video](demo.gif)
*The video shows the detections of a Faster R-CNN with a ResNet-50 backbone trained on the Sim10k dataset and tested on the Cityscapes dataset. The baseline version is displayed on the left, while OSSA is shown on the right. We see that OSSA is more effective at detecting cars.*



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

