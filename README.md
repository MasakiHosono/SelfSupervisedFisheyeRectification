# SelfSupervisedFiseyeRectification
An official pytorch implementation of the paper "Self-Supervised Fisheye Image Rectification by Reconstructing Coordinate Relations"

![results.png](https://raw.githubusercontent.com/MasakiHosono/SelfSupervisedFisheyeRectification/main/statics/results.png?token=AE3JGTNHNWMGTDYSXFK7PHDAOVLIQ "results.png")

Our network is based on single parameter division model, architecture is shown below.

![net_arch_full.png](https://raw.githubusercontent.com/MasakiHosono/SelfSupervisedFisheyeRectification/main/statics/net_arch_full.png?token=AE3JGTIG7EIOR2BT5B2DDMDAOVLLC "net_arch_full.png")

### Quick start
1. Clone the project.
   ```
   git clone https://github.com/MasakiHosono/SelfSupervisedFisheyeRectification.git
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
```
@inproceedings{hosono2021self,
  title={Self-Supervised Deep Fisheye Image Rectification Approach using Coordinate Relations},
  author={Hosono, Masaki and Simo-Serra, Edgar and Sonoda, Tomonari},
  booktitle={2021 17th International Conference on Machine Vision and Applications (MVA)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```
