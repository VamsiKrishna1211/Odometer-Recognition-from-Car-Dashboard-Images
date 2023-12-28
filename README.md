# Odometer Recognition from Car Dashboard Images (ClearQuote Assessment | Exercise - 2)

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
- Initially the model is trained for 20 Epochs, then the best model is saved and then the model is trained for another 20 epochs with lower learning rates than the previous one.
- The model is trained on a batch size of 8.
#### Training Results
- The model is trained for 40 epochs in total.
- The model is trained on a batch size of 8.
- For Multiple taining run graphs, refer to the tensorboard logs `tb_logs/fasterrcnn_v2` folder. 
    - Use the command `tensorboard --logdir tb_logs/fasterrcnn_v2` to view the tensorboard logs in a browser.
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
- The model is trained on a batch size of 16.
- The model is trained for 100 epochs.
#### Training Results
- The model is trained for 100 epochs in total.
- The model is trained on a batch size of 16.
- For Multiple taining run graphs, refer to the tensorboard logs `tb_logs/trocr_v2_expanded` folder. 
    - Use the command `tensorboard --logdir tb_logs/trocr_v2_expanded` to view the tensorboard logs in a browser.
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

## Environment Setup
### Using Conda
- Create a new conda environment using the command `conda create --name <env_name> python=3.9.18`
- Activate the environment using the command `conda activate <env_name>`
- Install the required packages using the command `pip install -r requirements.txt`
