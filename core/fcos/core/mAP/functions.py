from .classes import MapMaster
import torch

def fcos_to_boxes(classes, confs, locs, centers, row, col):
    iou_lime = 0.5  # threshold for iou
    cls_lime = 0.2  # threshold for confidence
    
    # obtain the size of all the feature maps
    map_sizes = []
    for map_num in range(len(confs)):
        # obtain the size of the feature map
        H = confs[map_num].size(2)
        W = confs[map_num].size(3)
        map_sizes.append([H, W])
    # initialize a manager for feature maps
    map_master = MapMaster(map_sizes)
    
    # initialize a list for storing predicted bounding boxes of different classes
    GTmaster = []
    for i in classes:
        GTmaster.append([])
    
    # traverse all feature maps
    for feature_num in range(len(confs)):
        conf = confs[feature_num].detach().cpu()
        loc = locs[feature_num].detach().cpu()
        center = centers[feature_num].detach().cpu()
        # suppress confidence
        conf = conf * center
        # obtain non-background area
        indexes = torch.max(conf, 1)[1]
        indexes = indexes.numpy().tolist()[0]
        # search for pixels on the feature map whose confidence are over threshold
        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                # the pixel is considered as positive sample if its confidence is larger than the threshold
                if conf[0, indexes[i][j], i, j] >= cls_lime:
                    box = [feature_num, i, j, indexes[i][j], conf[0, indexes[i][j], i, j], loc[0, 0, i, j],
                           loc[0, 1, i, j], loc[0, 2, i, j], loc[0, 3, i, j]]
                    box = map_master.decode_coordinate(box, row, col)
                    GTmaster[indexes[i][j]].append(box)
    # initialize a empty list for returning the final detected bounding boxes after NMS
    boxes = []
    # non maximum suppression (NMS)
    for GT in GTmaster:
        while len(GT) > 0:
            max_obj = []
            for obj in GT[:]:
                # obtain the bounding box with the highest confidence  within the same category
                if max_obj == []:
                    max_obj = obj
                    continue
                if max_obj[1] < obj[1]:
                    max_obj = obj
            GT.remove(max_obj)
            # select the bounding box of the highest confidence as a final predicted box
            boxes.append(max_obj)
            if len(GT) > 0:
                # remove other boxes of the same category whose iou between it and the selected box is larger than the threshold
                for obj in GT[:]:
                    # calculate the iou between it and the selected bounding box
                    iou = compute_iou([obj[2], obj[3], obj[4], obj[5]],
                                         [max_obj[2], max_obj[3], max_obj[4], max_obj[5]])
                    if iou > iou_lime:
                        # delete it when the iou breaks the threshold
                        GT.remove(obj)
    return boxes

def compute_iou(rect1, rect2):
    # calculate iou between two bounding boxes

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
    s_rect1 = (bottom1 - top1) * (right1 - left1)
    s_rect2 = (bottom2 - top2) * (right2 - left2)
    # calculate the coordinate of intersection rectangle
    cross_left = max(left1, left2)
    cross_right = min(right1, right2)
    cross_top = max(top1, top2)
    cross_bottom = min(bottom1, bottom2)
    # judge if the intersection exists
    if cross_left >= cross_right or cross_top >= cross_bottom:
        # the intersection does not exist
        return 0
    else:
        # the intersection exists
        # calculate the area of the intersection
        s_cross = (cross_right - cross_left) * (cross_bottom - cross_top)
        # return iou
        return s_cross / (s_rect1 + s_rect2 - s_cross)

def take2(elem):
    return elem[1]


def compute_mAP(points, GT_num):
    boxes_num = 0
    TP_num = 0
    AP = 0
    precision = 0
    recall = 0
    for point in points:
        boxes_num += 1
        if point[2] == True:
            TP_num += 1
            precision = TP_num / boxes_num
            AP += (1 / GT_num) * precision
        else:
            precision = TP_num / boxes_num
        recall = TP_num / GT_num
    return precision, recall, AP

def calculate_mAP_gt(box, gt_box,threshold=0.5):
    TP = False  # judge if the bndbox is True Positive

    iou = compute_iou(gt_box[1:5], box[2:6])
    if iou >= threshold:
        TP = True
    return (*box[:2], TP)

def return_mAP(model, dataset, classes):
    model.cuda()
    model.eval()
    
    mAP_all = {c:[] for c in classes}
    gt_count_all = {c:0 for c in classes}

    # traverse dataset
    for image, tags in dataset:
        # get an image
        cuda_image = image.unsqueeze(0).cuda()
        # predict
        row = cuda_image.shape[2]
        col = cuda_image.shape[3]
        confs, locs, centers = model(cuda_image)
        boxes = fcos_to_boxes(classes, confs, locs, centers, row, col)
        for gt_box in tags:
            box_class = classes[int(gt_box[0].item())]
            gt_count_all[box_class] += 1
            for box in boxes:
                if(box[0] == gt_box[0]):
                    mAP_all[box_class].append(calculate_mAP_gt(box, gt_box))
    mAP = 0
    mp = 0
    mr = 0
    for c in classes:
        p, r, ap = compute_mAP(mAP_all[c], gt_count_all[c])
        mAP += ap * gt_count_all[c]
        mp += p * gt_count_all[c]
        mr += r * gt_count_all[c]
    mAP /= sum(gt_count_all.values())
    mp /= sum(gt_count_all.values())
    mr /= sum(gt_count_all.values())
    return mAP, mp, mr
