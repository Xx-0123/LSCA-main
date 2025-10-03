# Enhancing Object Detection with Shape-IoU and Scale-Space-Task Collaborative Lightweight Path Aggregation
## Introduction
This repository contains source code for LSCA (Scale-Space-Task Collaborative Lightweight Path Aggregation) implemented with PyTorch. Object detection, a pivotal task in computer vision, aims to localize and recognize objects in images or videos. While deep learning-based approaches have shown superior robustness over traditional methods, they still face challenges in effectively integrating scale, spatial, and task information. To address these issues, we propose a novel algorithm that incorporates a Lightweight Path Aggregation Feature Pyramid Network (LPAFPN) enhanced with a Scale-Space-Task collaborative module (ALPAFPN) and a Shape-IoU loss function. The LPAFPN reduces model parameters by shuffling and fusing features across channels. Furthermore, the ALPAFPN module is integrated to boost its perception ability for joint information processing. Experimental results on the Pascal VOC and VisDrone2019-DET datasets demonstrate that our approach outperforms state-of-the-art algorithms in F1 score, precision, and mean average precision.
## Environments
To run the code in this repository, the following environment setup is recommended:
* Python 3.8+
* PyTorch 1.7.0+
* CUDA 10.2+ (for GPU acceleration)
* OpenCV-Python
* NumPy
* Pandas
* TorchVision
* yaml

You can install the required packages using pip:

    pip install torch torchvision opencv-python numpy pandas pyyaml
## Data Preparation
The code supports training and evaluation on the following datasets:<br>[VisDrone2019-DET](https://aistudio.baidu.com/datasetdetail/54054)<br>[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
### 1. VisDrone2019-DET Dataset
After downloading, organize the dataset into the following structure:

VisDrone2019-DET/<br>
├── train/<br>
│   ├── images/<br>
│   └── annotations/<br>
├── val/<br>
│   ├── images/<br>
│   └── annotations/<br>
└── test/<br>
    └── images/

### 2. Pascal VOC 2012 Dataset
The dataset should be organized in the standard Pascal VOC structure:

VOCdevkit/<br>
└── VOC2012/<br>
    ├── JPEGImages/<br>
    ├── Annotations/<br>
    ├── ImageSets/<br>
    │   └── Main/<br>
    └── ...

Modify the dataset configuration files (e.g., `data/visdrone.yaml` , `data/voc.yaml`) to point to the correct paths of your downloaded datasets.
## Run Experiments
### Training

To train the LSCA model, use the train_aux.py script. Example commands:<br>1. Train on VisDrone2019-DET dataset:
    
    python train_aux.py --data data/visdrone.yaml --cfg cfg/baseline/yolor-csp.yaml --weights '' --batch-size 16 --epochs 100 --name lsca_visdrone
<br>2. Train on Pascal VOC 2012 dataset:

    python train_aux.py --data data/voc.yaml --cfg cfg/deploy/yolov7x.yaml --weights '' --batch-size 16 --epochs 100 --name lsca_voc

Key parameters:

* `--data`: Path to the dataset configuration file
* `--cfg`: Path to the model configuration file
* `--weights`: Path to pre-trained weights (use '' for training from scratch)
* `--batch-size`: Batch size for training
* `--epochs`: Number of training epochs
* `--name`: Name of the experiment (for saving results)

### Inference
To perform inference with a trained model, use the detect.py script. Example command:<br>python detect.py --source inference/images/ --weights runs/train/lsca_visdrone/weights/best.pt --img-size 640 --conf-thres 0.25 --iou-thres 0.45

Key parameters:

* `--source`: Path to input images/videos or webcam (use 0 for webcam)
* `--weights`: Path to trained model weights
* `--img-size`: Inference image size
* `--conf-thres`: Object confidence threshold
* `--iou-thres`: IOU threshold for NMS (Non-Maximum Suppression)

### Evaluation
Model evaluation (e.g., mAP calculation) is automatically performed during training on the validation set. To run a separate evaluation, you can modify the training script or use the validation functionality integrated into the training pipeline.

## Notes
* The hyperparameters for training can be adjusted in the `hyp.scratch.p6.yaml` file.
* Model configurations (network architecture) are defined in the `cfg/` directory.
* Training results, including weights, logs, and evaluation metrics, will be saved in the `runs/` directory.

For more details on the implementation and algorithm, refer to the corresponding research paper (to be added).

    
