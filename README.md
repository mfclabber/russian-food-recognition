<!-- <div align="center">
  !NB This project is in development
</div>

*** -->

# Russian Food Recognition

This repository provides a robust framework for recognizing various Russian dishes using the Faster R-CNN deep learning architecture. This project aims to assist in the automatic identification and categorization of Russian cuisine in images, which can be applied in diverse domains such as food blogging, dietary tracking, and restaurant automation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Parameters](#parameters)

## Overview

The Russian Food Recognition project leverages the Faster R-CNN model to detect and classify different types of Russian food in images. Faster R-CNN is a state-of-the-art object detection model that provides high accuracy and speed, making it suitable for real-time food recognition applications.

<p align="center">
  <img src=./assets/sample_1.png/>
</p>

## Features

- **High Accuracy:** Utilizes Faster R-CNN for precise food detection and classification.
- **Extensive Dataset:** Trained on a diverse dataset of Russian dishes.
- **Scalability:** Easily adaptable to include more food categories or different cuisines.
- **Modular Design:** Clear separation of data processing, model training, and evaluation modules.

## Parameters

|                     Modules                      | Parameters |
|--------------------------------------------------|------------|
|         model.backbone.body.conv1.weight         |    9408    |
|    model.backbone.body.layer1.0.conv1.weight     |    4096    |
|    model.backbone.body.layer1.0.conv2.weight     |   36864    |
|    model.backbone.body.layer1.0.conv3.weight     |   16384    |
| model.backbone.body.layer1.0.downsample.0.weight |   16384    |
|    model.backbone.body.layer1.1.conv1.weight     |   16384    |
|    model.backbone.body.layer1.1.conv2.weight     |   36864    |
|    model.backbone.body.layer1.1.conv3.weight     |   16384    |
|    model.backbone.body.layer1.2.conv1.weight     |   16384    |
|    model.backbone.body.layer1.2.conv2.weight     |   36864    |
|    model.backbone.body.layer1.2.conv3.weight     |   16384    |
|    model.backbone.body.layer2.0.conv1.weight     |   32768    |
|    model.backbone.body.layer2.0.conv2.weight     |   147456   |
|    model.backbone.body.layer2.0.conv3.weight     |   65536    |
| model.backbone.body.layer2.0.downsample.0.weight |   131072   |
|    model.backbone.body.layer2.1.conv1.weight     |   65536    |
|    model.backbone.body.layer2.1.conv2.weight     |   147456   |
|    model.backbone.body.layer2.1.conv3.weight     |   65536    |
|    model.backbone.body.layer2.2.conv1.weight     |   65536    |
|    model.backbone.body.layer2.2.conv2.weight     |   147456   |
|    model.backbone.body.layer2.2.conv3.weight     |   65536    |
|    model.backbone.body.layer2.3.conv1.weight     |   65536    |
|    model.backbone.body.layer2.3.conv2.weight     |   147456   |
|    model.backbone.body.layer2.3.conv3.weight     |   65536    |
|    model.backbone.body.layer3.0.conv1.weight     |   131072   |
|    model.backbone.body.layer3.0.conv2.weight     |   589824   |
|    model.backbone.body.layer3.0.conv3.weight     |   262144   |
| model.backbone.body.layer3.0.downsample.0.weight |   524288   |
|    model.backbone.body.layer3.1.conv1.weight     |   262144   |
|    model.backbone.body.layer3.1.conv2.weight     |   589824   |
|    model.backbone.body.layer3.1.conv3.weight     |   262144   |
|    model.backbone.body.layer3.2.conv1.weight     |   262144   |
|    model.backbone.body.layer3.2.conv2.weight     |   589824   |
|    model.backbone.body.layer3.2.conv3.weight     |   262144   |
|    model.backbone.body.layer3.3.conv1.weight     |   262144   |
|    model.backbone.body.layer3.3.conv2.weight     |   589824   |
|    model.backbone.body.layer3.3.conv3.weight     |   262144   |
|    model.backbone.body.layer3.4.conv1.weight     |   262144   |
|    model.backbone.body.layer3.4.conv2.weight     |   589824   |
|    model.backbone.body.layer3.4.conv3.weight     |   262144   |
|    model.backbone.body.layer3.5.conv1.weight     |   262144   |
|    model.backbone.body.layer3.5.conv2.weight     |   589824   |
|    model.backbone.body.layer3.5.conv3.weight     |   262144   |
|    model.backbone.body.layer4.0.conv1.weight     |   524288   |
|    model.backbone.body.layer4.0.conv2.weight     |  2359296   |
|    model.backbone.body.layer4.0.conv3.weight     |  1048576   |
| model.backbone.body.layer4.0.downsample.0.weight |  2097152   |
|    model.backbone.body.layer4.1.conv1.weight     |  1048576   |
|    model.backbone.body.layer4.1.conv2.weight     |  2359296   |
|    model.backbone.body.layer4.1.conv3.weight     |  1048576   |
|    model.backbone.body.layer4.2.conv1.weight     |  1048576   |
|    model.backbone.body.layer4.2.conv2.weight     |  2359296   |
|    model.backbone.body.layer4.2.conv3.weight     |  1048576   |
|    model.backbone.fpn.inner_blocks.0.0.weight    |   65536    |
|     model.backbone.fpn.inner_blocks.0.0.bias     |    256     |
|    model.backbone.fpn.inner_blocks.1.0.weight    |   131072   |
|     model.backbone.fpn.inner_blocks.1.0.bias     |    256     |
|    model.backbone.fpn.inner_blocks.2.0.weight    |   262144   |
|     model.backbone.fpn.inner_blocks.2.0.bias     |    256     |
|    model.backbone.fpn.inner_blocks.3.0.weight    |   524288   |
|     model.backbone.fpn.inner_blocks.3.0.bias     |    256     |
|    model.backbone.fpn.layer_blocks.0.0.weight    |   589824   |
|     model.backbone.fpn.layer_blocks.0.0.bias     |    256     |
|    model.backbone.fpn.layer_blocks.1.0.weight    |   589824   |
|     model.backbone.fpn.layer_blocks.1.0.bias     |    256     |
|    model.backbone.fpn.layer_blocks.2.0.weight    |   589824   |
|     model.backbone.fpn.layer_blocks.2.0.bias     |    256     |
|    model.backbone.fpn.layer_blocks.3.0.weight    |   589824   |
|     model.backbone.fpn.layer_blocks.3.0.bias     |    256     |
|          model.rpn.head.conv.0.0.weight          |   589824   |
|           model.rpn.head.conv.0.0.bias           |    256     |
|         model.rpn.head.cls_logits.weight         |    768     |
|          model.rpn.head.cls_logits.bias          |     3      |
|         model.rpn.head.bbox_pred.weight          |    3072    |
|          model.rpn.head.bbox_pred.bias           |     12     |
|       model.roi_heads.box_head.fc6.weight        |  12845056  |
|        model.roi_heads.box_head.fc6.bias         |    1024    |
|       model.roi_heads.box_head.fc7.weight        |  1048576   |
|        model.roi_heads.box_head.fc7.bias         |    1024    |
|  model.roi_heads.box_predictor.cls_score.weight  |   132096   |
|   model.roi_heads.box_predictor.cls_score.bias   |    129     |
|  model.roi_heads.box_predictor.bbox_pred.weight  |   528384   |
|   model.roi_heads.box_predictor.bbox_pred.bias   |    516     |

**Total Trainable Params: 41 950 036**


<!-- ## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/mfclabber/russian-food-recognition.git
    cd russian-food-recognition
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the pretrained weights (if available) and place them in the `weights` directory. -->

<!-- ## Usage

To recognize Russian food in an image, use the provided script:

```sh
python recognize_food.py --image path_to_image.jpg
```

This will output the image with detected food items highlighted and classified. -->

## Dataset

The dataset on [HF](https://huggingface.co/datasets/mllab/alfafood) and on [Kaggle](https://www.kaggle.com/datasets/mfclabber/alfafood) in compressed image format used for training consists of a variety of images representing different Russian dishes. Each image is annotated with bounding boxes and labels corresponding to the food items present.

<!-- ## Training

To train the Faster R-CNN model on your dataset:

1. Prepare your dataset following the structure required by Faster R-CNN.
2. Configure the training parameters in `config.py`.
3. Run the training script:
    ```sh
    python train.py
    ```

## Evaluation

To evaluate the performance of the model on a test dataset, use the evaluation script:

```sh
python evaluate.py --test-data path_to_test_data
```

This will provide metrics such as precision, recall, and mean Average Precision (mAP). -->

# TODO:
- [ ] Make the project modular
- [ ] Finetuning YOLOv10, DERT
- [ ] Rewrite file with annotations
- [ ] To configure TensorBoard
