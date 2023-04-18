import argparse
import os

from PIL import Image
from transformers import pipeline

# argparse
parser = argparse.ArgumentParser(description='Infer Model')

parser.add_argument('--image_path', type=str, default=None,
                    help='single image inference')
parser.add_argument('--model_dir', type=str, default='./checkpoints/vit-large-91',
                    help='path to pretrained model directory')

parser.add_argument('--batch_images_dir', type=str, default=None,
                    help='batch images inference')

args = parser.parse_args()

# hyperparameter

MODEL_DIR = args.model_dir

IMAGE_PATH = args.image_path

BATCH_IMAGES_DIR = args.batch_images_dir

IDX2LABEL = {'LABEL_0': 'vasc',
             'LABEL_1': 'bcc',
             'LABEL_2': 'mel',
             'LABEL_3': 'nv',
             'LABEL_4': 'df',
             'LABEL_5': 'akiec',
             'LABEL_6': 'bkl',
             'LABEL_7': 'not a cancer image'}

# Build Pipline

assert not (IMAGE_PATH is None and BATCH_IMAGES_DIR is None), "Invalid Image Input."

if __name__ == '__main__':
    classifier = pipeline("image-classification", model=MODEL_DIR)

    if IMAGE_PATH is not None:
        img = Image.open(IMAGE_PATH)

        infer_result = classifier(img)

        img.close()
        print(f"Image: {IMAGE_PATH} Result: {IDX2LABEL[infer_result[0]['label']]}")
        print(infer_result)

    if BATCH_IMAGES_DIR is not None:
        files = os.listdir(BATCH_IMAGES_DIR)
        lens = len(files)
        imgs = []
        for i in range(lens):
            imgs.append(Image.open(BATCH_IMAGES_DIR + '\\' + files[i]))
        for i in range(lens):
            infer_result = classifier(imgs[i])
            print(f"Image: {files[i]} Result: {IDX2LABEL[infer_result[0]['label']]} Confi: {infer_result[0]['score']}")
            print(infer_result[:2])
