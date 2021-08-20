# SimpleNet
This is an ultra-lightweight human pose estimation network（#named SimpleNet) that can be run on the mobile terminal (iOS + Android) in real time.
The code and readme part are referenced in [1].

## Main Results
1.Qualitative results of validation set

（1）val_pred

![val_100_pred](https://user-images.githubusercontent.com/45363047/120256364-85d88100-c2c0-11eb-843d-d232d683044a.jpg)
（2）val_hm_pred

![val_100_hm_pred](https://user-images.githubusercontent.com/45363047/120256381-912bac80-c2c0-11eb-8f4a-cedcafe58945.jpg)

2.Quantitative results of validation set

![图片](https://user-images.githubusercontent.com/45363047/120257086-1fecf900-c2c2-11eb-9ce4-d613d82319cd.png)

3.Calculation amount

![图片](https://user-images.githubusercontent.com/45363047/120256753-6b52d780-c2c1-11eb-8e30-d7033c3e61c2.png)

4.forward Time(CPU_单线程) = 0.058


##Note:
Flip test is used.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. 
The code is developed and tested using GTX1050. 

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
3. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo) and caffe-style pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1yJMSFOnmzwhA4YYQS71Uy7X1Kl_xq9fN?usp=sharing). 
4. Download mpii and coco pretrained models from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW0D5ZE4ArK9wk_fvw) or [GoogleDrive](https://drive.google.com/drive/folders/13_wJ6nC7my1KKouMkQMqyr9r1ZnLnukP?usp=sharing). Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet50-caffe.pth.tar
            |   |-- resnet101-5d3b4d8f.pth
            |   |-- resnet101-caffe.pth.tar
            |   |-- resnet152-b121ed2d.pth
            |   `-- resnet152-caffe.pth.tar
            |-- pose_coco
            |   |-- pose_resnet_101_256x192.pth.tar
            |   |-- pose_resnet_101_384x288.pth.tar
            |   |-- pose_resnet_152_256x192.pth.tar
            |   |-- pose_resnet_152_384x288.pth.tar
            |   |-- pose_resnet_50_256x192.pth.tar
            |   `-- pose_resnet_50_384x288.pth.tar
            `-- pose_mpii
                |-- pose_resnet_101_256x256.pth.tar
                |-- pose_resnet_101_384x384.pth.tar
                |-- pose_resnet_152_256x256.pth.tar
                |-- pose_resnet_152_384x384.pth.tar
                |-- pose_resnet_50_256x256.pth.tar
                `-- pose_resnet_50_384x384.pth.tar

   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Valid on MPII using pretrained models of use our [Super light SimpleNet]

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/mobile/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/mobile/model_best.pth.tar

```

### Training on MPII

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

### Valid on COCO val2017 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
```

### Training on COCO train2017 use our [Super light SimpleNet]

```
python pose_estimation/train.py \
    --cfg experiments/coco/mobile/256x192_d256x3_adam_lr1e-3.yaml

```
### Citation
The code and readme part are referenced in
```
[1] Xiao B ,  Wu H ,  Wei Y . Simple Baselines for Human Pose Estimation and Tracking[C]// ECCV. arXiv, 2018.
```
