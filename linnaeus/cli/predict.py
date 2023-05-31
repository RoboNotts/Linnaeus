from linnaeus.core.models import FCOS
from linnaeus.core.loaders import ClassLoader
from linnaeus.core.data_augmentation import preprocessing
from linnaeus.core.mAP.functions import fcos_to_boxes
import numpy as np
import torch
import cv2

def main(model, weights, classfile, image):
    # load class list
    
    classes = ClassLoader(classfile)
        
    # load the model
    model = FCOS(len(classes),torch.load(model))
    model.load_state_dict(torch.load(weights))
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(train_device)
    model.eval()

    # obtain the path of the image
    frame = cv2.imread(str(image.resolve()))

    # torch digestive
    mcvities = preprocessing(torch.from_numpy(np.transpose(frame, (2, 0, 1)))).unsqueeze(0)
    mcvities = mcvities.to(train_device)

    row = frame.shape[0]
    col = frame.shape[1]
    # predict
    confs, locs, centers = model(mcvities)
    boxes = fcos_to_boxes(classes, confs, locs, centers, row, col)
    for box in boxes:
        xmin = box[2] * col // 480
        ymin = box[3] * row // 360
        xmax = box[4] * col // 480
        ymax = box[5] * row // 360
        # draw rectangle
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 40, 255), 2)
        frame = cv2.putText(frame, classes[box[0]] + ":" + str(round(box[1].item(), 2)), (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 40, 255), 1)
        
    cv2.imshow(f'WOW!', frame)
    cv2.waitKey(0) & 0xFF == ord('q')