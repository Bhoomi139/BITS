import os
from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from data_loader.dataset import data


def load_state_dict(model_dir, is_multi_gpu):
    state = torch.load(
        model_dir,
        map_location=lambda storage, loc: storage
    )["state_dict"]

    if is_multi_gpu:
        new_state_dict = OrderedDict()

        for k, v in state.items():
            name = k[7:]
            new_state_dict[name] = v

        return new_state_dict

    return state


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
    
    print("Starting Training")
    
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./runs", exist_ok=True)

    logger = Logger(
        "./logs/{}.log".format(args.model),
        len(args.resume) != 0
    )

    logger.append(vars(args))

    writer = SummaryWriter(
        log_dir="./runs/{}".format(args.model)
    ) if args.display else None

    gpus = args.gpu.split(",")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    print("Loading datasets...")

    train_dataset = data(
        image_dir=os.path.join(args.data_root, "train"),
        csv_file=os.path.join(
            args.data_root,
            "gt_avg_train.csv"
        ),
        transform=train_transform
    )

    val_dataset = data(
        image_dir=os.path.join(args.data_root, "valid"),
        csv_file=os.path.join(
            args.data_root,
            "gt_avg_valid.csv"
        ),
        transform=val_transform
    )

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    if args.debug:

        x, y = next(iter(train_loader))
        logger.append([x.shape, y.shape])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    is_use_cuda = torch.cuda.is_available()

    cudnn.benchmark = True


    print("Loading Model")
    model = build_model(args.model)

    if is_use_cuda:

        print("Using GPU")

        if len(gpus) == 1:
            model = model.cuda()
        else:
            model = nn.DataParallel(model.cuda())

    else:
        print("Using CPU")

    print(f"Model: {args.model}")
    
    print("Model Loaded Successfully")

    loss_fn = [
        nn.SmoothL1Loss(beta=1.0)
    ]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

    start_epoch = 0
    num_epochs = 100

    print("Optimizer : Adam")
    print(f"Loss Function : {loss_fn[0].__class__.__name__}")
    print("LR        : {}".format(
        optimizer.param_groups[0]["lr"]
    ))

    trainer = Trainer(
        model=model,
        model_type=args.model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_schedule=scheduler,
        log_batchs=50,
        is_use_cuda=is_use_cuda,
        train_data_loader=train_loader,
        valid_data_loader=val_loader,
        metric=None,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        is_debug=args.debug,
        logger=logger,
        writer=writer
    )

    print("Starting Training Loop")
    trainer.fit()
    logger.append("Training Finished Successfully.")
    if writer is not None:
        writer.close()

    
    print("Training Complete")
    print(f"Best model : ./checkpoint/{args.model}/best_model.ckpt")
    print(f"Training log : ./logs/{args.model}.log")
    print(f"History CSV : ./checkpoint/{args.model}/history.csv")

    if args.display:
        print(f"TensorBoard : ./runs/{args.model}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Age prediction"
    )

    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        help="Resume from checkpoint"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "-g",
        "--gpu",
        default="0",
        type=str,
        help="GPU ids (e.g. 0 or 0,1)"
    )

    parser.add_argument(
        "-d",
        "--data_root",
        default="./dataset",
        type=str,
        help="Dataset root directory"
    )

 parser.add_argument(
    "-m",
    "--model",
    default="resnet34-cbam",
    choices=[
        "resnet18-cbam",
        "resnet34-cbam",
        "resnet50-cbam",
        "resnet101-cbam",
        "resnet152-cbam"
    ],
    help="CBAM ResNet architecture"
)

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Training batch size"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable TensorBoard logging"
    )

    args = parser.parse_args()

    main(args)
    
