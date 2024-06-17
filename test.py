import sys
sys.path.append('SEGM-model/')
sys.path.append('OCR-model/')
sys.path.append('ReadingPipeline/')

import cv2
from matplotlib import pyplot as plt
import numpy as np
import json

from huggingface_hub import hf_hub_download

from ocrpipeline.predictor import PipelinePredictor
from ocrpipeline.linefinder import get_structured_text


def get_config_and_download_weights(repo_id, device='cpu'):
    # download weights and configs
    pipeline_config_path = hf_hub_download(repo_id, "pipeline_config.json")
    ocr_model_path = hf_hub_download(repo_id, "ocr/ocr_model.ckpt")
    ocr_config_path = hf_hub_download(repo_id, "ocr/ocr_config.json")
    segm_model_path = hf_hub_download(repo_id, "segm/segm_model.ckpt")
    segm_config_path = hf_hub_download(repo_id, "segm/segm_config.json")

    # change paths to downloaded weights and configs in main pipeline_config
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = json.load(f)

    pipeline_config['main_process']['SegmPrediction']['model_path'] = segm_model_path
    pipeline_config['main_process']['SegmPrediction']['config_path'] = segm_config_path
    pipeline_config['main_process']['SegmPrediction']['num_threads'] = 4
    pipeline_config['main_process']['SegmPrediction']['device'] = device
    pipeline_config['main_process']['SegmPrediction']['runtime'] = "Pytorch"

    pipeline_config['main_process']['OCRPrediction']['model_path'] = ocr_model_path
    pipeline_config['main_process']['OCRPrediction']['config_path'] = ocr_config_path
    pipeline_config['main_process']['OCRPrediction']['num_threads'] = 4
    pipeline_config['main_process']['OCRPrediction']['device'] = device
    pipeline_config['main_process']['OCRPrediction']['runtime'] = "Pytorch"

    # save pipeline_config
    with open(pipeline_config_path, 'w') as f:
        json.dump(pipeline_config, f)

    return pipeline_config_path


PIPELINE_CONFIG_PATH = get_config_and_download_weights("sberbank-ai/ReadingPipeline-Peter")

predictor = PipelinePredictor(pipeline_config_path=PIPELINE_CONFIG_PATH)

img_path = hf_hub_download("sberbank-ai/ReadingPipeline-Peter", "0_0.jpg")
image = cv2.imread(img_path)
rotated_image, pred_data = predictor(image)

structured_text = get_structured_text(pred_data, ['shrinked_text'])

result_text = [' '.join(line_text) for page_text in structured_text
                for line_text in page_text]

with open('out.txt', 'w') as f:
    f.write('\n'.join(result_text) + '\n')
