## Environment Setup (Linux or Windows)
> Note: The following steps are for setting the environment for inference only. <br>
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
    > - Download the zip file from the following link: [Google Drive](https://drive.google.com/file/d/1f56K0_etroH6bY5bwGfseDLFBahGTHxI/view?usp=drive_link) <br>
    > - Extract the zip file in the root directory of the project. <br>
    > - The extraction will create a file named `fasterrcnn-resnet50.pt` in the root directory of the project. <br>
    > - Create a folder named `weights/fasterrcnn` in the root directory of the project. <br>
    > - Place the downloaded weights (`fasterrcnn-resnet50.pt`) file in the `weights/fasterrcnn` folder. <br>
    > - The folder structure should look like this: <br>
    > ```bash
    > weights
    > ├── fasterrcnn
    >     └── fasterrcnn-resnet50.pt
    - **TrOCR** Model weights setup:
    > - Download the zip file from the following link: [Google Drive](https://drive.google.com/file/d/1Dwiiz-qS_bvMuLWXozs8DQQz4snN11OJ/view?usp=drive_link) <br>
    > - Extract the zip file in the root directory of the project. <br>
    > - The extraction will create a folder named `trocr_base_v2` in the root directory of the project. <br>
    > Copy the folder `trocr_base_v2` to the `weights/` folder. <br>
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
    >     ├── vocab.json

### Dataset
- The dataset can be downloaded or setup any way you want. There are no restrictions on the dataset folder structure.

## Running the code for Inference
- Please use the `test_predict.py` file to run the inference code.
- Following is the example command to run the inference code:
```bash
python test_predict.py --dataset-path dataset/test --image-extensions=".jpg,.png" --odometer-detector-weights weights/fasterrcnn/fasterrcnn-resnet50.pt --ocr-weights weights/trocr_base_v2 --output-path output.csv --save-images --save-images-path=save_images --device cuda
```
- The above command will run the inference code on the test set and save the results in the `output.csv` file.
- The `--save-images` flag will save the images with the predicted bounding boxes and the predicted text drawn onto the images in the `save_images` folder. (`--save-images-path`  flag needs to be specified for custom folder path.)
- The `--device` flag can be used to specify the device to run the inference on. The default value is `cuda`.
- The `--image-extensions` flag can be used to specify the image extensions to look for in the dataset folder. The default value is `".jpg,.png"`.
- The `--output-path` flag can be used to specify the output file path. The default value is `output.csv`.
- The `--save-images-path` flag can be used to specify the folder path to save the images with the predicted bounding boxes and the predicted text. The default value is `save_images`.
- The `--odometer-detector-weights` flag can be used to specify the path to the FasterRCNN model weights. The default value is `weights/fasterrcnn/fasterrcnn-resnet50.pt`.
- The `--ocr-weights` flag can be used to specify the path to the OCR model weights. The default value is `weights/trocr_base_v2`.
- The `--dataset-path` flag can be used to specify the path to the dataset folder. The default value is `dataset/test`.
- Use the command `python test_predict.py --help` to view the detailed help message.

## Environment Setup (Docker)
> Note: The following steps are for setting the environment for inference only. <br>

## Weights, Dataset setup.
### Weights
- Download the weights from the links provided above.
    - **FasterRCNN** Model weights setup:
    > - Download the zip file from the following link: [Google Drive](https://drive.google.com/file/d/1f56K0_etroH6bY5bwGfseDLFBahGTHxI/view?usp=drive_link) <br>
    > - Extract the zip file in the root directory of the project. <br>
    > - The extraction will create a file named `fasterrcnn-resnet50.pt` in the root directory of the project. <br>
    > - Create a folder named `weights/fasterrcnn` in the root directory of the project. <br>
    > - Place the downloaded weights (`fasterrcnn-resnet50.pt`) file in the `weights/fasterrcnn` folder. <br>
    > - The folder structure should look like this: <br>
    > ```bash
    > weights
    > ├── fasterrcnn
    >     └── fasterrcnn-resnet50.pt
    - **TrOCR** Model weights setup:
    > - Download the zip file from the following link: [Google Drive](https://drive.google.com/file/d/1Dwiiz-qS_bvMuLWXozs8DQQz4snN11OJ/view?usp=drive_link) <br>
    > - Extract the zip file in the root directory of the project. <br>
    > - The extraction will create a folder named `trocr_base_v2` in the root directory of the project. <br>
    > Copy the folder `trocr_base_v2` to the `weights/` folder. <br>
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
    >     ├── vocab.json

### Dataset
- The dataset can be downloaded or setup any way you want. There are no restrictions on the dataset folder structure.


## Docker Setup
- Install Docker using the following command:
> For clear instructions, refer to the official docker documentation [here](https://docs.docker.com/engine/install/ubuntu/)
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
- Verify the installation using the following command:
```bash
sudo docker run hello-world
```

## Docker Image Setup
- Build the docker image using the following command:
```bash
sudo docker build --file gpu.Dockerfile -t clearquote:latest .
```
> - The above command will build the docker image using the `gpu.Dockerfile` file and will tag the image as `clearquote:latest`. <br>
> - The `gpu.Dockerfile` file is used to build the docker image with GPU support. <br>
> - The `cpu.Dockerfile` file is used to build the docker image with CPU support. and replace `gpu.Dockerfile` with `cpu.Dockerfile` for only CPU supported docker image <br>

- Verify the docker image using the following command:
```bash
sudo docker images
```
- Run the docker image using the following command:
```bash
sudo docker run --rm -it --gpus all -v <absolute-path-to-dataset>:/clearquote/dataset -v <absolute-path-to-weights>:/clearquote/weights clearquote:latest bash
```
> - The above command will run the docker image and will open a bash shell inside the docker container. <br>

## Running the code for Inference
- Run the inference code using the following command inside the docker container:
```bash
python test_predict.py --dataset-path dataset/test --image-extensions=".jpg,.png" --odometer-detector-weights weights/fasterrcnn/fasterrcnn-resnet50.pt --ocr-weights weights/trocr_base_v2 --output-path output.csv --save-images --save-images-path=save_images --device cuda
```
> - The above command will run the inference code on the test set and save the results in the `output.csv` file. <br>
> - The `--save-images` flag will save the images with the predicted bounding boxes and the predicted text drawn onto the images in the `save_images` folder. (`--save-images-path`  flag needs to be specified for custom folder path.) <br>
> - The `--device` flag can be used to specify the device to run the inference on. (choices: cpu, cuda) The default value is `cuda`. <br>
