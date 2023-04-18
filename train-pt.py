import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize, RandomResizedCrop
from tqdm import tqdm
from vit_pytorch import ViT

# hyperparameter

IMAGES_DIR = "./archive/HAM10000_images/"
METADATA_CSV = "./archive/HAM4000_metadata.csv"
CHECKPOINTS = "./checkpoints/"
LOG_STEP = 3
SAVE_PER_EPOCH_NUM = 2
IS_PARALLEL = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 2
DEVICES = [torch.device(f"cuda:{i}") for i in range(min(torch.cuda.device_count(), NUM_GPUS))]

IMAGE_SIZE = 384
PATCH_SIZE = 16

SPLIT = 0.8
EPOCH = 40
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

if IS_PARALLEL:
    DEVICE = DEVICES[0]
    BATCH_SIZE *= len(DEVICES)
    LEARNING_RATE *= len(DEVICES)


# utils functions and classes

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]).convert("RGB") as img:

            if self.transform:
                return self.transform(img), self.labels[idx]
            else:
                return img, self.labels[idx]


def ReadImage(dataframe: pd.DataFrame, images_path: str) -> list:
    """image_id: str -> PIL.Image"""
    assert "image_id" in dataframe.columns and "dx" in dataframe.columns

    lens, _ = dataframe.shape
    postfix = ".jpg"
    dataset = []
    t = tqdm(range(lens))
    for i in t:
        t.set_description("Reading Image")
        dataset.append(
            {
                "image": images_path + dataframe['image_id'][i] + postfix,
                "label": torch.tensor(LABEL2IDX[dataframe['dx'][i]])
            }
        )
    return dataset


# load dataset

# IDX2LABEL = dict(enumerate(set(df['dx'])))
IDX2LABEL = {0: 'vasc',
             1: 'bcc',
             2: 'mel',
             3: 'nv',
             4: 'df',
             5: 'akiec',
             6: 'bkl'}

# LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}
LABEL2IDX = {'vasc': 0,
             'bcc': 1,
             'mel': 2,
             'nv': 3,
             'df': 4,
             'akiec': 5,
             'bkl': 6}

df = pd.read_csv(METADATA_CSV)

ds = ReadImage(df, IMAGES_DIR)

# split

random.seed(1919810)
random.shuffle(ds)

lens = len(ds)
train_ds = ds[:int(lens * SPLIT)]
dev_ds = ds[int(lens * SPLIT):]

print(f"Train : {len(train_ds)}")
print(f"DEV : {len(dev_ds)}")

# train_ds = ImageDataset(**{
#     'images': samp["image"],
#     'labels': samp["label"], } for samp in train_ds)

# train_ds = ImageDataset(*[(samp["image"], samp["label"]) for samp in train_ds])

transform = Compose([
    # Resize(IMAGE_SIZE),
    RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
    Normalize(0, 1),
])

train_ds = ImageDataset(images=[samp["image"] for samp in train_ds],
                        labels=[samp["label"] for samp in train_ds],
                        transform=transform)

dev_ds = ImageDataset(images=[samp["image"] for samp in dev_ds],
                      labels=[samp["label"] for samp in dev_ds],
                      transform=transform)

train_dl = DataLoader(dataset=train_ds,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=0)
dev_dl = DataLoader(dataset=dev_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=0)

# define model

model = ViT(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=len(IDX2LABEL),
    dim=2048,
    depth=16,
    heads=20,
    mlp_dim=4096,
    dropout=0.1,
    emb_dropout=0
).to(DEVICE)

if IS_PARALLEL:
    model = nn.DataParallel(model, device_ids=DEVICES)

if __name__ == '__main__':
    # optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    epoch = tqdm(range(EPOCH))
    for e in epoch:
        epoch.set_description(f"Epoch {e}")
        model.train()
        for batch_idx, (images, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images).to(DEVICE)
            # out = torch.argmax(out, dim=-1)
            loss = F.cross_entropy(out, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % LOG_STEP == 0:
                print('Train Epoch: {} [{}/{} ({:.3f}%)] Lr: {:e} Loss: {:.6f}'.format(
                    e, batch_idx, len(train_dl),
                    100. * batch_idx / len(train_dl), optimizer.param_groups[0]['lr'], loss.item()))
        dev_loss = []
        model.eval()
        with torch.no_grad():
            acc = []

            for batch_idx, (images, labels) in enumerate(dev_dl):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                out = model(images).to(DEVICE)

                pred = torch.argmax(out, dim=-1)

                acc += [1 if pred[i] == labels[i] else 0 for i in range(len(pred))]

                loss = F.cross_entropy(out, labels)
                dev_loss.append(loss)
            acc = sum(acc) / len(acc)

            dev_loss = torch.mean(torch.tensor(dev_loss))

            print("\nEpoch {} Validation loss: {:.6f} Accuracy: {:.6f}%.\n".format(e, dev_loss, acc * 100))

        if e % SAVE_PER_EPOCH_NUM == 0 or e == EPOCH - 1:
            save_file = "VIT-large-{}px-{}patch-{}epoch-{:.4f}loss.bin".format(IMAGE_SIZE, PATCH_SIZE, e, dev_loss)
            torch.save(model.state_dict(), CHECKPOINTS + save_file)
            print(f"Save to {CHECKPOINTS + save_file}\n")
