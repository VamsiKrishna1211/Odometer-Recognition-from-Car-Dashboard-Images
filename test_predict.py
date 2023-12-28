#! /usr/bin/env python3

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision import ops as tv_ops
import numpy as np
from pathlib import Path
from PIL import  Image
import sys
import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.roi_heads import fastrcnn_loss

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import numpy as np
from pathlib import Path
import logging
import sys
import albumentations as A
from fastprogress.fastprogress import master_bar, progress_bar
from typing import List, Optional, Union, Tuple, Dict, Any
import pandas as pd

import argparse
import warnings

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("trainer")
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
# logger.setLevel(logging.DEBUG)

class NoOdometerFoundException(Exception):
    message = "No odometer found in the image"

class OdometerOCRException(Exception):
    message = "Error in OCR"

class OdometerNoTextException(Exception):
    message = "No text found in the image"


class FasterRCNNModel(object):
    def __init__(self, weights, num_classes, device, transform=None):
        self.weights = weights
        self.num_classes = num_classes
        self.device = device

        self.model = None
        # self.model = self.build_model()
        # self.model.eval().to(self.device)

        self.transform = transform

    def load_model(self, weights=None):
        
        logger.debug("Loading FasterRCNNModel")
        if weights is None:
            if self.weights is None:
                raise Exception("Weights not provided")
            else:
                weights = self.weights
        
        if isinstance(weights, str) or isinstance(weights, Path):
            state_dict = torch.load(weights)
        else:
            state_dict = weights

        logger.debug("Loading FasterRCNNModel weights")
        model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=self.num_classes)
        model.load_state_dict(state_dict)
        logger.debug("Successfully loaded FasterRCNNModel weights")

        logger.debug("Loading Model to device {}".format(self.device))
        self.model = model
        self.model.eval().to(self.device)
        logger.debug("Successfully loaded FasterRCNNModel")


        return model
    
    def revert_transform(self, image, boxes, labels, shape):
        if self.transform is None:
            return image, boxes
        else:
            transform = A.Compose([
                    A.Resize(shape[0], shape[1]),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

            boxes = np.ceil(boxes).astype(np.int32)

            if isinstance(boxes, np.ndarray):
                boxes = boxes.tolist()
            elif isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy().tolist()
            elif isinstance(boxes, list):
                pass
            else:
                raise Exception("Invalid type for boxes")
            output = transform(image=image, bboxes=boxes, labels=labels)
            bboxes = output["bboxes"]
        
        return output["image"], bboxes
            
    
    def predict(self, image, class_num=2, iou_threshold=0.5, score_threshold=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if self.model is None:
            raise Exception("Model not loaded")

        with torch.no_grad():
            original_immage_shape = image.shape
            image = self.transform(image=image)['image']
            # image = image.unsqueeze(0)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1).to(self.device)/255.
            image = image.unsqueeze(0)

            with torch.no_grad():
                pred = self.model(image)
            pred = pred[0]

            nms_indexes = tv_ops.batched_nms(pred["boxes"], pred["scores"], pred["labels"], iou_threshold=iou_threshold)

            bboxes = torch.index_select(input=pred["boxes"], index=nms_indexes, dim=0)
            labels = torch.index_select(input=pred["labels"], index=nms_indexes, dim=0)
            scores = torch.index_select(input=pred["scores"], index=nms_indexes, dim=0)

            bboxes = torch.index_select(bboxes, index=torch.where(labels == class_num)[0], dim=0)
            labels = torch.index_select(labels, index=torch.where(labels == class_num)[0], dim=0)
            scores = torch.index_select(scores, index=torch.where(labels == class_num)[0], dim=0)

            bboxes = torch.index_select(bboxes, index=torch.where(scores > 0.5)[0], dim=0)
            labels = torch.index_select(labels, index=torch.where(scores > 0.5)[0], dim=0)
            scores = torch.index_select(scores, index=torch.where(scores > 0.5)[0], dim=0)

            pred_boxes = bboxes.cpu().numpy()
            pred_scores = scores.cpu().numpy()
            pred_labels = labels.cpu().numpy()

            # _, pred_boxes = self.revert_transform(image.cpu().numpy(), pred_boxes, pred_labels, original_immage_shape)
            
            
        return pred_boxes, pred_scores, pred_labels
        

class TrOCRModel(object):
    def __init__(self, weights=None, device=None, hf_model_name=None):
        self.weights = weights
        self.device = device
        self.hf_model_name = hf_model_name

        if self.hf_model_name is None:
            raise Exception("hf_model_name not provided")

        self.model: VisionEncoderDecoderModel = None
        self.processor: TrOCRProcessor = None

        self.transform = A.Compose([A.Resize(384, 384),])

        # self.model = self.build_model()
        # self.model.eval().to(self.device)

    def load_model(self, weights=None):

        logger.debug("Loading TrOCRModel and transferring to device {}".format(self.device))
        
        model = VisionEncoderDecoderModel.from_pretrained(self.hf_model_name).to(self.device)
        model.bfloat16()
        processor = TrOCRProcessor.from_pretrained(self.hf_model_name)
        logger.debug("Model Weights type: {}".format(model.dtype))

        model = torch.compile(model)
        # model.half()
        model.eval()

        logger.debug("Successfully loaded TrOCRModel")

        # logger.debug("Loading TrOCRModel weights")
        # model.load_state_dict(state_dict)
        

        self.model = model
        self.processor = processor

        logger.debug("TrOCRModel loaded")

        return model, processor
    
    def postprocess_text(self, text: str):
        text = ''.join(filter(str.isdigit, text))
        return text
    
    def predict(self, image) -> str:
            
        if self.model is None:
            raise Exception("Model not loaded")
        
        if self.processor is None:
            raise Exception("Processor not loaded")

        with torch.no_grad():
            image = np.array(image)
            image = self.transform(image=image)['image']
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            # pixel_values.half()
            # pixel_values = pixel_values.unsqueeze(0)

            # logger.debug(f"{pixel_values.shape}")

            generated_ids = self.model.generate(pixel_values, max_new_tokens=100)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # logger.debug(f"Generated text: {generated_text}")
            generated_text = self.postprocess_text(generated_text)

        
        return generated_text

            
    
class Inference(object):
    def __init__(self, device):
        self.device = device
        
        self.detection_model: FasterRCNNModel = None
        self.ocr_model: TrOCRModel = None

    def load_detection_model(self, weights, num_classes, transform=None):
        self.detection_model = FasterRCNNModel(weights, num_classes, self.device, transform)
        self.detection_model.load_model()

    def load_ocr_model(self, hf_model_name=None):
        self.ocr_model = TrOCRModel(device=self.device, hf_model_name=hf_model_name)
        self.ocr_model.load_model()

    def expand_bbox(self, bbox, scale=1.0):
        x1, y1, x2, y2 = bbox
        x1 = x1 - (x2 - x1) * (scale - 1) / 2
        x2 = x2 + (x2 - x1) * (scale - 1) / 2
        y1 = y1 - (y2 - y1) * (scale - 1) / 2
        y2 = y2 + (y2 - y1) * (scale - 1) / 2

        return x1, y1, x2, y2

    def predict(self, image: Image, class_num=2, iou_threshold=0.5, score_threshold=0.5):
        
        image = image.convert('RGB')
        image = image.resize((1280, 720))

        image_array = np.array(image)

        pred_boxes, pred_scores, pred_labels = self.detection_model.predict(image_array, class_num, iou_threshold, score_threshold)

        if len(pred_boxes) == 0:
            raise NoOdometerFoundException()

        # highest_score_index = np.argmax(pred_scores)
        
        # pred_boxes = pred_boxes[highest_score_index]
        # pred_scores = pred_scores[highest_score_index]
        # pred_labels = pred_labels[highest_score_index]

        for index, bbox in enumerate(pred_boxes):

            # bbox = pred_boxes
            bbox = self.expand_bbox(bbox, 1.2)


            odometer_image = image.crop(bbox)
            odometer_image = odometer_image.convert('RGB')

            pred_text = self.ocr_model.predict(odometer_image)

            if len(pred_text) > 0:
                break

        if len(pred_text) == 0:
            raise OdometerNoTextException()

        return bbox, pred_scores[index], pred_labels[index], pred_text

def draw_image(image, boxes, texts, scores, save_path=None):

    image = np.array(image)
    image = cv2.resize(image, (1280, 720))

    boxes = np.array(boxes).astype(np.int32)
    cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0))
    cv2.putText(image, "{}: {}".format(texts, scores), (boxes[0], boxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if save_path is not None:
        cv2.imwrite(str(save_path), image)
    else:
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        logger.warning("Save path not provided, image will not be saved")

def main(args):
    pass

    logger.setLevel(args.log_level)

    transform = A.Compose([
                    A.Resize(720, 1280),
                ])

    inference = Inference(args.device)
    inference.load_detection_model(args.odometer_detector_weights, num_classes=3, transform=transform)
    inference.load_ocr_model(hf_model_name=args.ocr_weights)

    image_extensions = args.image_extensions
    if image_extensions is None:
        image_extensions = ["jpg"]
        logger.info("No image extensions provided, using default extensions: {}".format(image_extensions))
    
    image_extensions = [x.lower() for x in image_extensions]

    image_paths = []
    for ext in image_extensions:
        ext = ext.replace(".", "")
        for path in Path(args.dataset_path).glob("**/*.{}".format(ext)):
            image_paths.append(path)
    
    logger.info("Found {} images".format(len(image_paths)))
    if len(image_paths) == 0:
        raise Exception("No images found")
    

    pred_boxes, pred_scores, pred_labels, pred_texts = [], [], [], []
    
    no_info_count = 0
    errored = False
    for image_path in progress_bar(image_paths):
        errored = False
        image = Image.open(image_path)
        try:
            pred_box, pred_score, pred_label, pred_text = inference.predict(image)
        except NoOdometerFoundException:
            logger.info("Image: {}, No odometer found".format(image_path))
            pred_box, pred_score, pred_label, pred_text = "", 0, "", ""
            no_info_count += 1
            errored = True

        except OdometerOCRException:
            logger.info("Image: {}, Error in OCR".format(image_path))
            pred_box, pred_score, pred_label, pred_text = "", 0, "", ""
            no_info_count += 1
            errored = True

        except OdometerNoTextException:
            logger.info("Image: {}, No text found".format(image_path))
            pred_box, pred_score, pred_label, pred_text = "", 0, "", ""
            no_info_count += 1
            errored = True

        except Exception as e:
            logger.error("Image: {}".format(image_path))
            raise e

        pred_boxes.append(pred_box)
        pred_scores.append(pred_score)
        pred_labels.append(pred_label)
        pred_texts.append(pred_text)

        if args.save_images and not errored:
            draw_image(image, pred_box, pred_text, pred_score, save_path=args.save_images_path/image_path.name)
    
    logger.info("No info found in {} images".format(no_info_count))
    logger.info("Saving predictions to {}".format(args.output_path))
    df = pd.DataFrame({
        "image_name": [x.name for x in image_paths],
        "pred_texts": pred_texts,
        "pred_scores": pred_scores,
        "pred_boxes": pred_boxes,
        "image_path": image_paths,

    })

    if args.output_path.endswith(".csv"):
        df.to_csv(args.output_path, index=False)
    elif args.output_path.endswith(".json"):
        df.to_json(args.output_path, orient="records")
    elif args.output_path.endswith(".pkl"):
        df.to_pickle(args.output_path)
    elif args.output_path.endswith(".xlsx"):
        df.to_excel(args.output_path, index=False)

    logger.info("Done")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--dataset-path', type=str, help='Path to images')
    parser.add_argument('--image-extensions', type=str, default="jpg", help='Image extensions, for multiple extensions, separate them by comma')
    parser.add_argument('--odometer-detector-weights', type=str, default="weights/fasterrcnn/fasterrcnn-resnet50.pt", help='Path to weights file of odometer detector')
    parser.add_argument('--ocr-weights', type=str, default="weights/trocr_base_v2", help="Path to the folder containing the weights of OCR model")
    parser.add_argument('--output-path', type=str, default="output", help="Path to the file where the output will be saved")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument("--save-images", action="store_true", help="Save images with bounding boxes and text")
    parser.add_argument("--save-images-path", type=str, default="save_images", help="Path to save images with bounding boxes and tex (if save-images is True) NOTE: This path should exist")
    args = parser.parse_args()

    # Validating the arguments
    if not os.path.exists(args.dataset_path):
        raise Exception("Dataset path does not exist")

    if not os.path.exists(args.odometer_detector_weights):
        raise Exception("Odometer detector weights path does not exist")
    
    if not os.path.exists(args.ocr_weights):
        raise Exception("OCR weights path does not exist")
    
    if args.image_extensions is not None:
        args.image_extensions = args.image_extensions.split(",")
    
    if os.path.exists(args.output_path):
        logger.warning("Output path already exists, the files will be overwritten")

    if args.device is None:
        logger.warning("Device not provided, using auto-detected device")
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device {}".format(args.device))
    elif args.device is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        elif args.device == "cpu":
            warnings.warn("Using CPU, this will be slow")
    
    if Path(args.output_path).suffix == ".csv":
        logger.info("Using CSV as output format")
    elif Path(args.output_path).suffix == ".json":
        logger.info("Using JSON as output format")
    elif Path(args.output_path).suffix == ".pkl":
        logger.info("Using Pickle as output format")
    elif Path(args.output_path).suffix == ".xlsx":
        logger.info("Using Excel as output format")
    else:
        raise Exception("Invalid output format, supported formats are: csv, json, pkl, xlsx")
    
    if args.save_images:
        if not os.path.exists(args.save_images_path):
            raise Exception("Save images path does not exist")
        else:
            args.save_images_path = Path(args.save_images_path)
    else:
        args.save_images_path = None

    main(args)
