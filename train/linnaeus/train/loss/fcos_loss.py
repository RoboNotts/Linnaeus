import torch
import torch.nn as nn
from linnaeus.core.mAP import MapMaster
from .iou_loss import IOULoss

class FCOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, confses, locses, centerses, labels, device):
        # location,confidence: results of forward propaganda[loc,conf]
        # labels: annotations
        # obtain batch_size
        batch_size = len(labels)
        # initialize iou loss
        iou_loss = IOULoss()
        # initialize BCE loss
        center_loss = nn.BCELoss()
        # initialize overall loss of the minibatch
        loss = torch.tensor(0).type('torch.FloatTensor').to(device)
        ########## calculate loss function for each image ##########
        for img_num in range(batch_size):
            # initialize loss of confidence
            loss_conf = torch.tensor(0).type('torch.FloatTensor').to(device)
            # initialize loss of location
            loss_l = torch.tensor(0).type('torch.FloatTensor').to(device)
            # initialize loss of offset
            loss_center = torch.tensor(0).type('torch.FloatTensor').to(device)
            # the label of current image
            tags = labels[img_num]
            # obtain all the feature maps
            confs = [confses[i][img_num] for i in range(5)]
            locs = [locses[i][img_num] for i in range(5)]
            centers = [centerses[i][img_num] for i in range(5)]
            # obtain the sizes of all feature maps
            map_sizes = []
            for map_num in range(len(confs)):
                # obtain feature map size
                H = confs[map_num].size(1)
                W = confs[map_num].size(2)
                map_sizes.append([H, W])
            # initialize a manager of feature maps
            map_master = MapMaster(map_sizes)
            # the number of positive samples
            poses = 0
            for feature_map in map_master.feature_maps:
                # obtain current feature map
                conf = confs[feature_map.num]
                loc = locs[feature_map.num]
                center = centers[feature_map.num]
                # calculate the loss of confidence
                conf = torch.clamp(conf, min=0.00000001, max=0.99999999)
                loss_c = - (1 - self.alpha) * conf ** self.gamma * torch.log(1 - conf)
                for pixel in feature_map.pixels:
                    pixel.judge_pos(tags)
                    # calculate loss function
                    # calculate the loss of confidence
                    for c_channel in range(conf.shape[0]):
                        if c_channel == pixel.tag[0]:
                            loss_c[c_channel, pixel.i, pixel.j] = (- self.alpha * (1 - conf[c_channel, pixel.i, pixel.j]) ** self.gamma) * torch.log(conf[c_channel, pixel.i, pixel.j])
                    if pixel.status == 1:
                        poses += 1
                        # calculate loss of location
                        loss_l = loss_l + iou_loss([pixel.tag[1], pixel.tag[2], pixel.tag[3], pixel.tag[4]],
                                                   [pixel.x - loc[0, pixel.i, pixel.j],
                                                    pixel.y - loc[1, pixel.i, pixel.j],
                                                    pixel.x + loc[2, pixel.i, pixel.j],
                                                    pixel.y + loc[3, pixel.i, pixel.j]])
                        # calculate loss of offset
                        loss_center = loss_center + center_loss(center[:, pixel.i, pixel.j],
                                                                torch.Tensor([pixel.center]).to(device))
                loss_conf += loss_c.sum()
            loss += loss_center + (loss_conf + loss_l) / poses if poses != 0 else loss_center + loss_conf + loss_l
#            if loss == - torch.log(torch.tensor(0).type('torch.FloatTensor')):
#                pass
        ###############################################################
        loss = loss / batch_size
        return loss
