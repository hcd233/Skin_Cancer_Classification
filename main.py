# import library
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


def compute_accuracy():
    correct = 0
    lens, _ = df.shape
    postfix = ".jpg"
    t = tqdm(range(lens))

    for i in t:
        t.set_description(f"Correct Num: {correct}")
        img = Image.open(IMAGES_DIR + df['image_id'][i] + postfix)
        res = classifier(img)
        if IDX2LABEL[res[0]['label']] == df['dx'][i]:
            correct += 1
    return correct / lens


IMAGES_DIR = './archive/HAM10000_images/'
METADATA_PATH = './archive/HAM10000_metadata.csv'
MODEL_DIR = './checkpoints/vit-large-91'

IDX2LABEL = {'LABEL_0': 'vasc',
             'LABEL_1': 'bcc',
             'LABEL_2': 'mel',
             'LABEL_3': 'nv',
             'LABEL_4': 'df',
             'LABEL_5': 'akiec',
             'LABEL_6': 'bkl'}

df = pd.read_csv(METADATA_PATH, usecols=['image_id', 'dx'])
classifier = pipeline("image-classification", model=MODEL_DIR)

if __name__ == '__main__':
    acc = compute_accuracy()
    print("Accuracy for entire dataset: {:.6f}".format(acc))
    # 94.0589 %
