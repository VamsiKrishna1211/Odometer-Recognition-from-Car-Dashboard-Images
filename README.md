# Odometer Recognition from Car Dashboard Images (ClearQuote Assessment | Exercise - 2)
## Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Final Inference](#final-inference)
- [Results](#results)
- [Environment Setup (Linux or Windows)](#environment-setup-linux-or-windows)
    - [Python Environment Setup](#python-environment-setup)
    - [Weights, Dataset setup.](#weights-dataset-setup)
        - [Weights](#weights)
        - [Dataset](#dataset)
    - [Running the code for Inference](#running-the-code-for-inference)

## Introduction
### Hardware & Software Used:
- **CPU:** Intel(R) Core(TM) i5-10900k CPU @ 3.70GHz
- **GPU:** NVIDIA GeForce RTX 3090
- **RAM:** 32GB
- **OS:** ArchLinux (Kernel: 6.1.68-1)
- **Python Version:** 3.9.18
- **PyTorch Version:** 2.1.2
- **CUDA Version:** 12.1

There are mainly four parts in this project:
- **Data Preprocessing**
- **Model Training**
- **Model Testing**
- **Final Inference**

## Data Preprocessing
The data preprocessing part is done in the explorer.ipynb notebook.
Basically the notebook is used to Explore the data and to find the best possible way to preprocess the data.
The notebook contains the following sections:
1. Load the data in VIA format
2. Convert the data to COCO format
3. Visualize the data
4. Separate the data into train and test sets.

Dataset Folder Structure:
```bash
dataset
├── odometer_expanded
│   ├── test
│   │   └── 62a501c62be4ea4a151632ba
│   ├── train
│   │   ├── 62a4ff852be4ea4a151632a7
│   │   ├── 62a4ff852be4ea4a151632a8
│   │   ├── 62a4ff862be4ea4a151632a9
│   │   ├── 62a4ff862be4ea4a151632aa
│   │   ├── 62a4ff862be4ea4a151632ab
│   │   ├── 62a4ff862be4ea4a151632ac
│   │   ├── 62a4ff862be4ea4a151632ad
│   │   ├── 62a4ff872be4ea4a151632af
│   │   ├── 62a4ff872be4ea4a151632b0
│   │   ├── 62a4ff872be4ea4a151632b2
│   │   ├── 62a4ff872be4ea4a151632b3
│   │   ├── 62a4ff872be4ea4a151632b4
│   │   ├── 62a4ff872be4ea4a151632b5
│   │   ├── 62a501c62be4ea4a151632b6
│   │   ├── 62a501c62be4ea4a151632b7
│   │   ├── 62a501c62be4ea4a151632b8
│   │   └── 62a501c62be4ea4a151632b9
│   └── val
│       ├── 62a501c72be4ea4a151632bb
│       └── 62a501c72be4ea4a151632bc
├── test
│   └── 62a501c62be4ea4a151632ba
├── train
│   ├── 62a4ff852be4ea4a151632a7
│   ├── 62a4ff852be4ea4a151632a8
│   ├── 62a4ff862be4ea4a151632a9
│   ├── 62a4ff862be4ea4a151632aa
│   ├── 62a4ff862be4ea4a151632ab
│   ├── 62a4ff862be4ea4a151632ac
│   ├── 62a4ff862be4ea4a151632ad
│   ├── 62a4ff872be4ea4a151632af
│   ├── 62a4ff872be4ea4a151632b0
│   ├── 62a4ff872be4ea4a151632b2
│   ├── 62a4ff872be4ea4a151632b3
│   ├── 62a4ff872be4ea4a151632b4
│   ├── 62a4ff872be4ea4a151632b5
│   ├── 62a501c62be4ea4a151632b6
│   ├── 62a501c62be4ea4a151632b7
│   ├── 62a501c62be4ea4a151632b8
│   └── 62a501c62be4ea4a151632b9
└── val
    ├── 62a501c72be4ea4a151632bb
    └── 62a501c72be4ea4a151632bc

```

## Model Training
The model training part is divided into two parts primarily:
1. Training One model to detect the odometer.
2. Training another model (OCR model) to detect the digits in the odometer.

### Training the odometer detection model
- **Model Used**: Faster-RCNN
- **Backbone**: Resnet50-FPN
- **Dataset Annotation format**: COCO
#### Training process
> Notebook used: `Trainer_v2.ipynb` <br>
> Dataset folders used: `dataset/train`, `dataset/val` and `dataset/test`
- The model is trained using the Faster-RCNN model with Resnet50 as the backbone.
- The model has 4 parts mainly:
    - The backbone
    - The backbone Feature Pyramid Network (FPN)
    - The Region Proposal Network (RPN)
    - The Region of Interest (RoI) Pooling
- The model is trained on different learning rates for different parts of the model and used SGD as the optimizer and OneCycleLR as the learning rate scheduler.
- Refer to the image [here](assets/example_learning_rate.png) for a sample OneCycleLR learning rate scheduler.
- Initially the model is trained for 20 Epochs, then the best model is saved and then the model is trained for another 20 epochs with lower learning rates than the previous one.
- The model is trained on a batch size of 8.
#### Training Results
- The model is trained for 40 epochs in total.
- The model is trained on a batch size of 8.
- For Multiple taining run graphs, refer to the tensorboard logs `tb_logs/fasterrcnn_v2` folder. 
    - Run the command `tensorboard --logdir tb_logs/fasterrcnn_v2` in the root folder to view the tensorboard logs in a browser.
    - **Note:** The checkpoints are not saves as the weights are too large to upload to github.
#### Model Testing
- The model is tested on the test set.
- The finalized model was tested on the test set, and visually verifying the results.

### Training the OCR model
- **Model Used**: TrOCR Base Model (Transformer based OCR model)
#### Training process
> Notebook used: `TrainerOCR.ipynb` <br>
> Dataset folders used: `dataset/odometer_expanded/train`, `dataset/odometer_expanded/val` and `dataset/odometer_expanded/test`
- The model is trained using the [TrOCR base model](https://huggingface.co/docs/transformers/model_doc/trocr).
- The model has 2 parts mainly:
    - The Encoder
        - The Encoder is a ViT Transformer Encoder.
    - The Decoder
        - The Decoder is a RoBERTa Transformer Decoder.
- The model is trained on different learning rates for different parts of the model and used SGD as the optimizer and OneCycleLR as the learning rate scheduler.
- Refer to the image [here](assets/example_learning_rate.png) for a sample OneCycleLR learning rate scheduler.
- The model is trained on a batch size of 16.
- The model is trained for 100 epochs.
#### Training Results
- The model is trained for 100 epochs in total.
- The model is trained on a batch size of 16.
- For Multiple taining run graphs, refer to the tensorboard logs `tb_logs/trocr_v2_expanded` folder. 
    - Run the command `tensorboard --logdir tb_logs/trocr_v2_expanded` in the root folder to view the tensorboard logs in a browser.
    - **Note:** The checkpoints are not saves as the weights are too large to upload to github.
#### Model Testing
- The model is tested on the test set. `dataset/odometer_expanded/test`
- The finalized model was tested on the test set, and visually verifying the results.

### Model Testing
- The two Models `fasterrcnn_v2` and `trocr_v2_expanded` are tied together to perform the final inference testing.
- The model is tested on the test set. `dataset/test`
- The finalized model was tested on the test set, and visually verifying the results.

### Final Inference
> Notebook used: `Inference.ipynb` <br>
> Dataset folders used: `dataset/odometer_expanded/test`
- The model is tested on the test set. `dataset/odometer_expanded/test`

## Results
### Odometer Detection Model
- The model is trained for 40 epochs in total.
- The model is trained on a batch size of 8.
- Final combined Loss (box_reg, classifier, objectness, rpn_box_reg) value: `0.106`
- Weights link: [Google Drive](https://drive.google.com/file/d/1-8Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z/view?usp=sharing)

### OCR Model
- The model is trained for 100 epochs in total.
- The model is trained on a batch size of 16.
- Final Loss value: `0.1176`
- Final CER (Character Error Rate) value: `0.0544`
- Weights link: [Google Drive](https://drive.google.com/file/d/1-8Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z/view?usp=sharing)

## Environment Setup (Linux or Windows)
## Python Environment Setup
### Using Conda (Recommended)
- Create a new conda environment using the command `conda create --name <env_name> python=3.9.18`
- Activate the environment using the command `conda activate <env_name>`
- Install the required packages using the command `pip install -r requirements/requirements.txt`
> - (**Not Recommended**) If you are using only CPU, then install the `requirements/requirements-cpu.txt` file instead of `requirements/requirements.txt` <br>
> - (**Not Recommended**) If you're using a windows system and do not have Visual Studio installed, then install the `requirements/requirements-bkup.txt` file instead of `requirements/requirements.txt` **Note:** This will install the CPU version of PyTorch.

### Using System Python & Pip (Linux Only)
- Install the `virtualenv` package using the command `pip install virtualenv`
- Create a new virtual environment using the command `virtualenv <env_name>`
 - Activate the environment using the command `source <env_name>/bin/activate`
- Install the required packages using the command `pip install -r requirements/requirements.txt`
> - (**Not Recommended**) If you are using only CPU, then install the `requirements/requirements-cpu.txt` file instead of `requirements/requirements.txt` <br>

## Weights, Dataset setup.
### Weights
- Download the weights from the links provided above.
    - **FasterRCNN** Model weights setup:
    > - Create a folder named `weights/fasterrcnn` in the root directory of the project. <br>
    > - Place the downloaded weights file in the `weights/fasterrcnn` folder. <br>
    > - The folder structure should look like this: <br>
    > ```bash
    > weights
    > ├── fasterrcnn
    >     └── fasterrcnn-resnet50.pt
    - **TrOCR** Model weights setup:
    > - Downlload the zip file from the following link: [Google Drive](https://drive.google.com/file/d/1-8Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z/view?usp=sharing) <br>
    > - Extract the zip file in the root directory of the project. <br>
    > Copy the folder `trocr_v2` to the `weights` folder. <br>
    > - The folder structure should look like this: <br>
    > ```bash
    > weights
    > ├── trocr_base_v2
    >     ├── config.json
    >     ├── generation_config.json
    >     ├── merges.txt
    >     ├── model.safetensors
    >     ├── preprocessor_config.json
    >     ├── special_tokens_map.json
    >     ├── tokenizer_config.json
    >     ├── tokenizer.json
    >     |── vocab.json

### Dataset
- The dataset can be downloaded or setup any way you want. There are no restrictions on the dataset folder structure.

## Running the code for Inference
- Please use the `test_predict.py` file to run the inference code.
- following is the example command to run the inference code:
```bash
python test_predict.py --dataset-path dataset/test --image-extensions=".jpg,.png" --odometer-detector-weights weights/fasterrcnn/fasterrcnn-resnet50.pt --ocr-weights weights/trocr_base_v2 --output-path output.xlsx --save-images --save-images-path=save_images --device cuda
```
- The above command will run the inference code on the test set and save the results in the `output.xlsx` file.
- The `--save-images` flag will save the images with the predicted bounding boxes and the predicted text drawn onto the images in the `save_images` folder. (`--save-images-path`  flag needs to be specified for custom folder path.)
- The `--device` flag can be used to specify the device to run the inference on. The default value is `cuda`.
- The `--image-extensions` flag can be used to specify the image extensions to look for in the dataset folder. The default value is `".jpg,.png"`.
- The `--output-path` flag can be used to specify the output file path. The default value is `output.xlsx`.
- The `--save-images-path` flag can be used to specify the folder path to save the images with the predicted bounding boxes and the predicted text. The default value is `save_images`.
- The `--odometer-detector-weights` flag can be used to specify the path to the FasterRCNN model weights. The default value is `weights/fasterrcnn/fasterrcnn-resnet50.pt`.
- The `--ocr-weights` flag can be used to specify the path to the OCR model weights. The default value is `weights/trocr_base_v2`.
- The `--dataset-path` flag can be used to specify the path to the dataset folder. The default value is `dataset/test`.
- Use the command `python test_predict.py --help` to view the help message.
