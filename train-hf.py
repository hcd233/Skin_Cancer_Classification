# import library
import argparse
import os
from random import shuffle, seed
import evaluate
import numpy as np
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, \
    DefaultDataCollator

# argparse
parser = argparse.ArgumentParser(description='Train Model')

parser.add_argument('--metadata_path', type=str, default='./archive/HAM10000_metadata.csv',
                    help='path to metadata file')
parser.add_argument('--images_dir', type=str, default='./archive/HAM10000_images/',
                    help='path to images directory')
parser.add_argument('--model_dir', type=str, default='../model/vit-large-patch16-224-in21k',
                    help='path to pretrained model directory')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                    help='path to save model checkpoints')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='ratio of warmup steps to total training steps')
parser.add_argument('--split', type=float, default=0.8, help='train-validation split ratio')
parser.add_argument('--gpu', type=str, default='0', help='CUDA visible devices')
parser.add_argument('--logging_steps', type=int, default='50', help='Print log per step')
args = parser.parse_args()

# hyperparameter

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

METADATA_PATH = args.metadata_path

IMAGES_DIR = args.images_dir

MODEL_DIR = args.model_dir

CHECKPOINTS_DIR = args.checkpoints_dir

LEARNING_RATE = args.learning_rate

BATCH_SIZE = args.batch_size

EPOCHS = args.epochs

WARMUP_RATIO = args.warmup_ratio

SPLIT = args.split

LOGGING_STEPS = args.logging_steps

RAW_PATH = "./archive/raw"


# utils functions

def ReadImage(dataframe: pd.DataFrame, images_path: str) -> list:
    """image_id: str -> PIL.Image"""

    assert "image_id" in dataframe.columns and "dx" in dataframe.columns

    lens, _ = dataframe.shape
    postfix = ".jpg"
    dataset = []
    t = tqdm(range(lens))
    for i in t:
        t.set_description("Reading Image")
        img = Image.open(images_path + dataframe['image_id'][i] + postfix)
        dataset.append(
            {
                "image": img,
                "label": dataframe['dx'][i]
            }
        )
        img.close()
    return dataset


def ReadRaw() -> list:
    imgs = os.listdir(RAW_PATH)
    dataset = []
    t = tqdm(imgs)
    for i in t:
        t.set_description("Reading Raw Image")
        img = Image.open(RAW_PATH + "/" + i)
        dataset.append(
            {
                "image": img,
                "label": 'not a cancer image'
            }
        )
        img.close()
    return dataset


def transforms(examples):
    trans = _transforms()
    examples["pixel_values"] = [trans(img.convert("RGB")) for img in examples["image"]]
    examples["label"] = [LABEL2IDX[label] for label in examples["label"]]
    del examples["image"]
    return examples


def _transforms():
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    return Compose([RandomResizedCrop(size), ToTensor(), normalize])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


# import dataset

# Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec),
# basal cell carcinoma (bcc),
# benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl),
# dermatofibroma (df),
# melanoma (mel),
# melanocytic nevi (nv),
# vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

# IDX2LABEL = dict(enumerate(set(df['dx'])))
IDX2LABEL = {0: 'vasc',
             1: 'bcc',
             2: 'mel',
             3: 'nv',
             4: 'df',
             5: 'akiec',
             6: 'bkl',
             7: 'not a cancer image'}

# LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}
LABEL2IDX = {'vasc': 0,
             'bcc': 1,
             'mel': 2,
             'nv': 3,
             'df': 4,
             'akiec': 5,
             'bkl': 6,
             'not a cancer image': 7}

if __name__ == '__main__':
    seed(114514)

    df = pd.read_csv(METADATA_PATH, usecols=['image_id', 'dx'])

    ds = ReadImage(dataframe=df, images_path=IMAGES_DIR)
    shuffle(ds)

    raw_ds = ReadRaw()
    shuffle(raw_ds)

    train_ds = ds[:int(SPLIT * len(ds))] + raw_ds[:int(SPLIT * len(raw_ds))]
    dev_ds = ds[int(SPLIT * len(ds)):] + raw_ds[int(SPLIT * len(raw_ds)):]

    ds = {
        "train": Dataset.from_list(train_ds),
        "dev": Dataset.from_list(dev_ds)
    }

    ds = DatasetDict(ds)

    # preprocess dataset

    image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    ds = ds.with_transform(transforms)
    data_collator = DefaultDataCollator()

    # define metric

    acc = evaluate.load("accuracy")

    # import model

    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR,
                                                            num_labels=len(IDX2LABEL),
                                                            ignore_mismatched_sizes=True)

    # train model

    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
