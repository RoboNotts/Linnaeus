import torch
from fcos.core.loaders import FolderDataSetLoader, ClassLoader
from fcos.train.loss import FCOSLoss
from fcos.core.models import FCOS
import torch.utils.data as Data
import os
from fcos.core.mAP import return_mAP
import torch
from tqdm import tqdm
from math import ceil

# torch.manual_seed(1)	#reproducible
def train(weights, classfile, train_dataset, val_dataset, batch_size = 1, epoch = 1000, lr = 0.0001, ft_lr = 0.000001, start=0, weight_decay = 0.005, optimizer_name="Adam", save_file=False):

    # initialize model
    model = FCOS(torch.load(weights))

    # initailize gpu for training and testing
    if torch.cuda.is_available():
        print("Using CUDA")
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(train_device)
    
    # apply hyper-parameters
    weight_p, bias_p, FT_weight_p, FT_bias_p, feat_weight_p, feat_bias_p = [], [], [], [], [], []

    for name, mp in model.named_parameters():
        if 'FT' in name:
            if 'bias' in name:
                FT_bias_p += [mp]
            else:
                FT_weight_p += [mp]
        else:
            if 'bias' in name:
                bias_p += [mp]
            else:
                weight_p += [mp]

    # initialize optimizer and loss function
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': weight_decay, 'lr': lr},
                                    {'params': bias_p, 'weight_decay': 0, 'lr': lr},
                                    {'params': FT_weight_p, 'weight_decay': weight_decay, 'lr': ft_lr},
                                    {'params': FT_bias_p, 'weight_decay': 0, 'lr': ft_lr},
                                    ], momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': weight_decay, 'lr': lr},
                                    {'params': bias_p, 'weight_decay': 0, 'lr': lr},
                                    {'params': FT_weight_p, 'weight_decay': weight_decay, 'lr': ft_lr},
                                    {'params': FT_bias_p, 'weight_decay': 0, 'lr': ft_lr},
                                    ])
    

    classes = ClassLoader(classfile)

    loss_func = FCOSLoss()
    # initialize training set
    train_dataset = FolderDataSetLoader(train_dataset, classes)
    val_dataset = FolderDataSetLoader(val_dataset, classes)
    # initialized dataloader
    loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # shuffle the dataset
        num_workers=4,  # reading dataset by multi threads
        collate_fn=FolderDataSetLoader.collate_fn
    )
    # initialize maximum mAP
    max_mAP = 0
    # training
    if not os.path.exists("module"):
        os.makedirs("module")
    with tqdm(total=100, position=2, desc="Accuracy") as tqdm_accuracy, tqdm(total=100, position=3, desc="Recall") as tqdm_recall, tqdm(total=100, position=4, desc="Mean Average Precision") as tqdm_mAP:
        for c_epoch in tqdm(range(start, epoch), position=0, desc="Epoch", leave=True):
            # release a mini-batch data
            with tqdm(enumerate(loader), total=ceil(len(train_dataset) / batch_size), unit_scale=batch_size, position=1, desc="Step") as tdqm_enumerated_loader:
                for step, (images, tags) in tdqm_enumerated_loader:
                    # read images and labels
                    device_image = images.to(train_device)
                    # obtain feature maps output by the model
                    confs, locs, centers = model(device_image)  # .to(train_device)
                    # training
                    loss = loss_func(confs, locs, centers, tags, train_device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tqdm.write('Step: %d | train loss: %.4f' % (step, loss))
                    tdqm_enumerated_loader.set_postfix(loss = '%.4f' % loss)
            
            # evaluate the performance of current model
            mAP, mp, mr = return_mAP(model, val_dataset, classes)
            tqdm_accuracy.reset()
            tqdm_accuracy.update(mp * 100)
            tqdm_recall.reset()
            tqdm_recall.update(mr * 100)
            tqdm_mAP.reset()
            tqdm_mAP.update(mAP * 100)
            tqdm.write('Epoch: %d | mAP: %.4f' % (c_epoch, mAP))
            # save if better
            if mAP >= max_mAP:
                if save_file:
                    torch.save(model.state_dict(), save_file)
                max_mAP = mAP
