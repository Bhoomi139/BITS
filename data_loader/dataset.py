import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class data(Dataset):

    def __init__(
        self,
        image_dir,
        csv_file,
        transform=None,
        target="apparent_age_avg"
    ):

        self.image_dir = image_dir
        self.transform = transform
        self.target = target

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["file_name"]
        face_filename = filename + "_face.jpg"
        image_path = os.path.join(self.image_dir, face_filename)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target == "apparent_age_avg":
            label = float(row["apparent_age_avg"])
        elif self.target == "real_age":
            label = float(row["real_age"])
        else:
            raise ValueError("target must be 'apparent_age_avg' or 'real_age'")

        label = torch.tensor(label, dtype=torch.float32)

        return image, label