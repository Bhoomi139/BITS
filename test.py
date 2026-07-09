import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr

import model.resnet_cbam as resnet_cbam
from data_loader.dataset import data


def build_model(model_name):

    if model_name == "resnet18-cbam":

        model = resnet_cbam.resnet18_cbam(
            pretrained=True
        )


    elif model_name == "resnet34-cbam":

        model = resnet_cbam.resnet34_cbam(
            pretrained=True
        )


    elif model_name == "resnet50-cbam":

        model = resnet_cbam.resnet50_cbam(
            pretrained=True
        )


    elif model_name == "resnet101-cbam":

        model = resnet_cbam.resnet101_cbam(
            pretrained=True
        )


    elif model_name == "resnet152-cbam":

        model = resnet_cbam.resnet152_cbam(
            pretrained=True
        )


    else:
        raise ValueError(
            "Unknown model {}".format(model_name)
        )


    
    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    return model


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Loading Test Dataset")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_dataset = data(
        image_dir=os.path.join(args.data_root, "test"),
        csv_file=os.path.join(args.data_root, "gt_avg_test.csv"),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test Images : {len(test_dataset)}")
    print("Loading Model")
    model = build_model(args.model)

    checkpoint = torch.load(
        args.weights,
        map_location=device
    )

    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)

    model.eval()

    criterion = nn.SmoothL1Loss(beta=1.0)

    predictions = []
    targets = []

    running_loss = 0.0
    total_samples = 0

    start = time.time()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device).float().squeeze()

            outputs = model(images).squeeze(1)

            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions.extend(
                outputs.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    inference_time = time.time() - start

    predictions = np.array(predictions)
    targets = np.array(targets)

    test_loss = running_loss / total_samples

    mae = mean_absolute_error(
        targets,
        predictions
    )

    mse = mean_squared_error(
        targets,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        targets,
        predictions
    )

    corr, _ = pearsonr(
        targets,
        predictions
    )

    print("\n")
    print("Test Results")
    print(f"Test Loss (Huber): {test_loss:.4f}")
    print(f"MAE              : {mae:.4f}")
    print(f"MSE              : {mse:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"R2 Score         : {r2:.4f}")
    print(f"Pearson Corr.    : {corr:.4f}")
    print(f"Test Images      : {len(test_dataset)}")
    print(f"Inference Time   : {inference_time:.2f} sec")

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame({
        "GroundTruth": targets,
        "Prediction": predictions
    })

    df.to_csv(
        "results/predictions.csv",
        index=False
    )

    print("Predictions saved to results/predictions.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default="./dataset",
        type=str
    )

    parser.add_argument(
        "--weights",
        default="./checkpoint/resnet34-cbam/best_model.ckpt"",
        type=str
    )

    parser.add_argument(
    "--model",
    default="resnet34-cbam",
    choices=[
        "resnet18-cbam",
        "resnet34-cbam",
        "resnet50-cbam",
        "resnet101-cbam",
        "resnet152-cbam"
    ]
)

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int
    )

    args = parser.parse_args()

    main(args) 
