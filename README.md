# SelfSupervisedFiseyeRectification
An official pytorch implementation of the paper "Self-Supervised Fisheye Image Rectification by Reconstructing Coordinate Relations"

### Quick start
1. Clone the project.
```
git clone https://github.com/MasakiHosono/SelfSupervisedFiseyeRectification.git
```
1. Install the dependencies.
```
pip3 install -r requirements.txt
```
1. Prepare dataset
Download [cityscapes dataset](https://www.cityscapes-dataset.com) and put it under `data/` in the project root.
Then run the following script.
```
python3 tools/prepare_cityscapes.py
```
1. Training.
```
python3 src/train.py cfg/cityscapes.yml
```
1. Testing.
```
python3 src/test.py cfg/cityscapes.yml
```

### Citation
TBD
