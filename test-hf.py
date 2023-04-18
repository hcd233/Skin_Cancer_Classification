import pandas as pd
from PIL import Image
from datasets import Dataset
from torch import device
from tqdm import tqdm
from transformers import pipeline

# hyper
TEST_IMAGES_DIR = "./archive/test/"
TEST_TRUTH_CSV = "./archive/test_truth.csv"
MODEL_PATH = "./checkpoints/vit-large-89-386"
BATCH_SIZE = 128
DEVICE = device("cuda:0")
SAVE_LOG = f"./logs/test-{MODEL_PATH[MODEL_PATH.rfind('/') + 1:]}.txt"


# utils functions

def ReadTestImage(dataframe: pd.DataFrame, prefix) -> list:
    assert "image" in dataframe.columns and "label" in dataframe.columns

    lens, _ = dataframe.shape
    postfix = ".jpg"
    dataset = []
    t = tqdm(range(lens))
    for i in t:
        t.set_description("Reading Image")
        img = Image.open(prefix + dataframe['image'][i] + postfix)
        dataset.append(
            {
                "name": dataframe['image'][i],
                "image": img,
                "label": dataframe['label'][i]
            }
        )
        img.close()
    return dataset


# csv
#       image    MEL     NV    BCC  AKIEC    BKL     DF   VASC
# 0     False  False   True  False  False  False  False  False
# 1     False  False   True  False  False  False  False  False
# 2     False  False  False  False  False   True  False  False
# 3     False  False   True  False  False  False  False  False
# 4     False  False   True  False  False  False  False  False
# ...     ...    ...    ...    ...    ...    ...    ...    ...
# 1507  False  False  False  False  False   True  False  False
# 1508  False  False   True  False  False  False  False  False
# 1509  False  False  False  False   True  False  False  False
# 1510  False  False  False  False  False   True  False  False
# 1511  False  False  False   True  False  False  False  False
#
# [1512 rows x 8 columns]


df = pd.read_csv(TEST_TRUTH_CSV)
df['label'] = df.iloc[:, 1:].idxmax(axis=1)
df = df.drop(columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])

# processed csv
#              image  label
# 0     ISIC_0034524     NV
# 1     ISIC_0034525     NV
# 2     ISIC_0034526    BKL
# 3     ISIC_0034527     NV
# 4     ISIC_0034528     NV
# ...            ...    ...
# 1507  ISIC_0036060    BKL
# 1508  ISIC_0036061     NV
# 1509  ISIC_0036062  AKIEC
# 1510  ISIC_0036063    BKL
# 1511  ISIC_0036064    BCC
#
# [1512 rows x 2 columns]

# dataset
IDX2LABEL = {'LABEL_0': 'VASC',
             'LABEL_1': 'BCC',
             'LABEL_2': 'MEL',
             'LABEL_3': 'NV',
             'LABEL_4': 'DF',
             'LABEL_5': 'AKIEC',
             'LABEL_6': 'BKL',
             'LABEL_7': 'Not A Cancer Image'}

ds = ReadTestImage(df, prefix=TEST_IMAGES_DIR)

ds = Dataset.from_list(ds)

# pineline

classifier = pipeline(task="image-classification",
                      model=MODEL_PATH,
                      device=DEVICE,
                      batch_size=BATCH_SIZE)

# test
t = tqdm(range(0, len(ds), BATCH_SIZE))
t.set_description("Testing")

infer_result = []
for i in t:
    batch_result = classifier(ds['image'][i:i + BATCH_SIZE])
    infer_result += batch_result

acc = sum([1 for i in range(len(infer_result)) if IDX2LABEL[infer_result[i][0]['label']] == ds['label'][i]]) / len(
    infer_result) \
      * 100

with open(SAVE_LOG, mode="w+", encoding="utf-8") as file:
    t = tqdm(range(len(infer_result)))
    t.set_description("Writing Result")
    file.write("Accuracy in {:d} samples : {:.6f}% \n".format(len(infer_result), acc))
    for i in t:
        file.write(f"IMAGE: {ds['name'][i]} "
                   f"Truth: {ds['label'][i]} "
                   f"Pred: {IDX2LABEL[infer_result[i][0]['label']]} "
                   f"Confi: {infer_result[i][0]['score']} \n")

print("Accuracy in {:d} samples : {:.6f}%".format(len(infer_result), acc))
