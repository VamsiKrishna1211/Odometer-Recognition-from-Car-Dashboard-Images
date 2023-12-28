Introduction
- Please refer to the Github Repository for detailed information: https://github.com/VamsiKrishna1211/Odometer-Recognition-from-Car-Dashboard-Images?tab=readme-ov-file
Hardware & Software Used:
- CPU: Intel(R) Core(TM) i5-10900k CPU @ 3.70GHz
- GPU: NVIDIA GeForce RTX 3090
- RAM: 32GB
- OS: ArchLinux (Kernel: 6.1.68-1)
- Python Version: 3.9.18
- PyTorch Version: 2.1.2
- CUDA Version: 12.1

There are mainly four parts in this project:
- Data Preprocessing
- Model Training
- Model Testing
- Final Inference

Data Preprocessing
The data preprocessing part is done in the explorer.ipynb notebook.
Basically, the notebook is used to explore the data and to find the best possible way to preprocess the data.
The notebook contains the following sections:
1. Load the data in VIA format
2. Convert the data to COCO format
3. Visualize the data
4. Separate the data into train and test sets.

Dataset Folder Structure:
dataset
├── odometer_expanded
│   ├── test
│   │   └── 62a501c62be4ea4a151632ba
│   ├── train
│   │   ├── 62a4ff852be4ea4a151632a7
│   │   ├── 62a4ff852be4ea4a151632a8
│   │   ├── 62a4ff862be4ea4a151632a9
│   │   ├── ....
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
│   ├── ....
└── val
    ├── 62a501c72be4ea4a151632bb
    └── 62a501c72be4ea4a151632bc


Model Training
The model training part is divided into two parts primarily:
1. Training One model to detect the odometer.
2. Training another model (OCR model) to detect the digits in the odometer.

Training the odometer detection model:
- Model Used: Faster-RCNN
- Backbone: Resnet50-FPN
- Dataset Annotation format: COCO
(Check the Github repository for detailed information about the model)


Training the OCR model:
- Model Used: TrOCR Base Model (Transformer based OCR model)
(Check the Github repository for detailed information about the model)


Model Testing
The two Models fasterrcnn_v2 and trocr_v2_expanded are tied together to perform the final inference testing.
The model is tested on the test set.

Final Inference
(Notebook used: Inference.ipynb)
The model is tested on the test set.

Results
Odometer Detection Model:
- The model is trained for 40 epochs in total.
- The model is trained on a batch size of 8.
- Final combined Loss value: 0.106
- Weights link: [Google Drive](https://drive.google.com/file/d/1f56K0_etroH6bY5bwGfseDLFBahGTHxI/view?usp=drive_link)

OCR Model:
- The model is trained for 100 epochs in total.
- The model is trained on a batch size of 16.
- Final Loss value: 0.1176
- Final CER (Character Error Rate) value: 0.0544
- Weights link: [Google Drive](https://drive.google.com/file/d/1Dwiiz-qS_bvMuLWXozs8DQQz4snN11OJ/view?usp=drive_link)
