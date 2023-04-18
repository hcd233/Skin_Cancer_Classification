import pandas as pd
import json
from tqdm import tqdm
from PIL import Image
import os
import shutil


labels_dir = "./archive/labels/"
imgs_dir = "./archive/HAM10000_images/"


def MakeLabels(dataframe: pd.DataFrame) -> list:
    """image_id: str -> PIL.Image"""

    assert "image_id" in dataframe.columns and "dx" in dataframe.columns

    lens, _ = dataframe.shape
    # postfix = ".jpg"
    dataset = []
    t = tqdm(range(lens))
    for i in t:
        t.set_description("Make labels")
        label_path = dataframe['image_id'][i] + ".json"
        img_path = dataframe['image_id'][i] + ".jpg"
        content = {"labels": [{"name": dataframe['dx'][i]}]}
        with open(labels_dir + label_path,mode='w+',encoding="utf-8") as file:
            json.dump(obj=content,fp=file,indent=4)
        shutil.copy(imgs_dir+img_path,labels_dir+img_path)

if __name__ == "__main__":
    df = pd.read_csv("./archive/HAM4000_metadata.csv")
    MakeLabels(df)