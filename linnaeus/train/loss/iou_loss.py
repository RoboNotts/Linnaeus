import torch
import torch.nn as nn

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, rect1, rect2):
        #calculate iou between two bounding boxes
        
        # obtain the index of boundaries
        left1 = rect1[0]
        top1 = rect1[1]
        right1 = rect1[2]
        bottom1 = rect1[3]
        left2 = rect2[0]
        top2 = rect2[1]
        right2 = rect2[2]
        bottom2 = rect2[3]
        # calculate the area of teh two bounding boxes, respectively
        s_rect1 = (bottom1 - top1 + 1) * (right1 - left1 + 1)
        s_rect2 = (bottom2 - top2 + 1) * (right2 - left2 + 1)
        # calculate the coordinate of intersection rectangle
        cross_left = max(left1, left2)
        cross_right = min(right1, right2)
        cross_top = max(top1, top2)
        cross_bottom = min(bottom1, bottom2)
        # judge if the intersection exists
        if cross_left >= cross_right or cross_top >= cross_bottom:
            # the intersection does not exist
            return torch.tensor(0).type('torch.FloatTensor')
        else:
            # the intersection exists
            # calculate the area of the intersection
            s_cross = (cross_right - cross_left + 1) * (cross_bottom - cross_top + 1)
            if s_rect1 + s_rect2 - s_cross <= 0 or s_cross <= 0:
                return torch.tensor(0).type('torch.FloatTensor')
            # return iou loss
            return - torch.log(s_cross / (s_rect1 + s_rect2 - s_cross))
