# Vision Transformer Detection
Repos for implementing Vision Transformer in PyTorch for Detection tasks:
- object detection
- instance segmentation

## ViT Architectures
Currently supporting:
- Fast/FasterRCNN
- Swin-based Transformer (work in progress)
## Datasets/Tasks
### Object Detection:
- COCO-2017
## Installation
- Install conda python
```bash
# python -m pip install -r requirements.txt

conda create -n openmmlab python=3.8 -y
conda activate openmmlab

conda install pytorch==1.9.0 torchvision cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.9.0/index.html

pip install -r requirements/build.txt
pip install -v -e .
```
```bash
conda create -n vision python=3.8 -y
conda activate vision

# conda install pytorch==1.8.0 torchvision cudatoolkit=10.1 -c pytorch -y
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html

conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install --upgrade pip==21.2.4
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install pandas numpy plotly nbformat==4.2 timm

pip install -r requirements/build.txt
pip install -v -e .

python tools/train.py configs/swin/swin_detect_backup.py

```
## Run - Object Detection
```bash
python tools/train.py configs/swin/swin_detect_backup.py
```
