# TraSw for ByteTrack

> [**TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**](https://arxiv.org/abs/2111.08954),            
> Delv Lin, Qi Chen, Chengyu Zhou, Kun He,              
> *[arXiv 2111.08954](https://arxiv.org/abs/2111.08954)*

**Related Works**

* [TraSw for FairMOT](https://github.com/DerryHub/FairMOT-attack)
* [TraSw for JDE](https://github.com/zhou0123/JDE-attack)

## Abstract

Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack.

## Attack Performance

**Single-Target Attack Results on MOT challenge test set**

| Dataset | Suc. Rate | Avg. Frames | Avg.  L<sub>2</sub> Distance |
| :-----: | :-------: | :---------: | :--------------------------: |
| 2DMOT15 |  91.38%   |    4.45     |             6.48             |
|  MOT17  |  92.91%   |    4.59     |             5.76             |

## Installation

* **same as** [ByteTrack](https://github.com/ifzhang/ByteTrack)

Step1. Install ByteTrack.

```
git clone https://github.com/DerryHub/ByteTrack-attack
cd ByteTrack-attack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```
pip3 install cython_bbox
```

## Data preparation

* 2DMOT15, MOT17 can be downloaded from the official webpage of [MOT-Challenge](https://motchallenge.net/). After downloading, you should prepare the data in the following structure:

  ```
  datasets
     ├── mot15
     │     ├── test
     │     └── train
     └── mot17
           ├── test
           └── train
  ```

* Then, you need to turn the datasets to COCO format and mix different training data:

  ```shell
  cd ByteTrack-attack
  python3 tools/convert_mot15_to_coco.py
  python3 tools/convert_mot17_to_coco.py
  python3 tools/convert_mot20_to_coco.py
  ```

## Target Model

* We choose bytetrack_x_mot17 [[google](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [baidu(code:ic0i)](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw)] trained by [ByteTrack](https://github.com/ifzhang/ByteTrack) as our primary target model.

## Tracking without Attack

```shell
python -m tools.track -f exps/example/mot/yolox_x_mix_det.py -c bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --img_dir datasets/mot15(mot17) --output_dir ${OUTPUT_DIR}
```

## Single-Target Attack

* attack all attackable objects separately in videos in parallel (may require a lot of memory).

```shell
python -m tools.track -f exps/example/mot/yolox_x_mix_det.py -c bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --img_dir datasets/mot15(mot17) --output_dir ${OUTPUT_DIR} --attack single --attack_id -1
```

* attack a specific object in a specific video (require to set specific video in `tools/convert_mot15/17_to_coco.py`).

```shell
python -m tools.track -f exps/example/mot/yolox_x_mix_det.py -c bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --img_dir datasets/mot15(mot17) --output_dir ${OUTPUT_DIR} --attack single --attack_id ${a specific id in origial tracklets}
```

## Acknowledgement

This source code is based on [ByteTrack](https://github.com/ifzhang/ByteTrack). Thanks for their wonderful works.

## Citation

```
@misc{lin2021trasw,
      title={TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking}, 
      author={Delv Lin and Qi Chen and Chengyu Zhou and Kun He},
      year={2021},
      eprint={2111.08954},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

