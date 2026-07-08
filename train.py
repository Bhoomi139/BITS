import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from data_loader.dataset import data 

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def main(args):
    print("Starting training")
    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('./logs/'+args.model+'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    print("Loading Data")

    train_datasets = data(image_dir=os.path.join(args.data_root, "train"),
                          csv_file=os.path.join(args.data_root, "gt_avg_train.csv"),
                          transform=data_transforms['train'])
    val_datasets   = data(image_dir=os.path.join(args.data_root, "valid"),
                          csv_file=os.path.join(args.data_root, "gt_avg_valid.csv"),
                          transform=data_transforms['val'])
    print(f"Training images:{len(train_datasets)}")
    print(f"Validation images:{len(val_datasets)}")
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size*len(gpus), shuffle=True, num_workers=4)
    val_dataloaders   = torch.utils.data.DataLoader(val_datasets, batch_size=1024, shuffle=False, num_workers=4)

    if args.debug:
        x, y =next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    print(f"Loading model: {args.model}")
    if  'resnet50' == args.model.split('_')[0]:
        my_model = models.resnet50(pretrained=True)
    elif 'resnet50-cbam' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50_cbam(pretrained=True)
    elif 'resnet101' == args.model.split('_')[0]:
        my_model = models.resnet101(pretrained=True)
    else:
        raise ModuleNotFoundError
    my_model.fc= nn.Linear(my_model.fc.in_features, 1)
    print("Loaded Model Succesfully")

    #my_model.apply(fc_init)
    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    loss_fn = [nn.L1Loss()]
    optimizer = optim.Adam(my_model.parameters(), lr=1e-4, weight_decay=1e-5) 
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)           #
    start_epoch = 0
    num_epochs  = 90
    
    my_trainer = Trainer(my_model, args.model, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_dataloaders, \
                        val_dataloaders, None, start_epoch, num_epochs, args.debug, logger, writer)
    print("Starting Training")
    my_trainer.fit()
    logger.append('Optimize Done!')
    print("Training completed successfully!")
    print("Best model saved in ./checkpoint/")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-d', '--data_root', default='./dataset',
                         type=str, help='data root')
    parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
                         type=str, help='train file')
    parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
                         type=str, help='validation file')
    parser.add_argument('-m', '--model', default='resnet50',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=12,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
