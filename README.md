<div align="center">
  !NB This project is in development
</div>

***

# Russian Food Recognition

This repository provides a robust framework for recognizing various Russian dishes using the Faster R-CNN deep learning architecture. This project aims to assist in the automatic identification and categorization of Russian cuisine in images, which can be applied in diverse domains such as food blogging, dietary tracking, and restaurant automation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)

## Overview

The Russian Food Recognition project leverages the Faster R-CNN model to detect and classify different types of Russian food in images. Faster R-CNN is a state-of-the-art object detection model that provides high accuracy and speed, making it suitable for real-time food recognition applications.

<p align="center">
  <img src=./assets/sample_4.png/>
</p>

## Features

- **High Accuracy:** Utilizes Faster R-CNN for precise food detection and classification.
- **Extensive Dataset:** Trained on a diverse dataset of Russian dishes.
- **Scalability:** Easily adaptable to include more food categories or different cuisines.
- **Modular Design:** Clear separation of data processing, model training, and evaluation modules.

## Installation

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

3. Download the pretrained weights (if available) and place them in the `weights` directory.

## Usage

To recognize Russian food in an image, use the provided script:

```sh
python recognize_food.py --image path_to_image.jpg
```

This will output the image with detected food items highlighted and classified.

## Dataset

The [dataset](https://huggingface.co/datasets/mllab/alfafood) used for training consists of a variety of images representing different Russian dishes. Each image is annotated with bounding boxes and labels corresponding to the food items present.

## Training

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

This will provide metrics such as precision, recall, and mean Average Precision (mAP).

# TODO:
- [ ] Make the project modular
- [ ] Finetuning YOLOv10, DERT
- [ ] Rewrite file with annotations
- [ ] To configure TensorBoard
