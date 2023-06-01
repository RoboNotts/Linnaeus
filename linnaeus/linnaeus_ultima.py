from linnaeus.core.models import FCOS
from linnaeus.core.loaders import ClassLoader
from linnaeus.core.data_augmentation import preprocessing
from linnaeus.core.mAP.functions import fcos_to_boxes
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_FCOS_MODEL = "weights.pt"
DEFAULT_RESNET_MODEL = "resnet50-19c8e357.pth"
DEFAULT_CLASSES = "classes.txt"

class LinnaeusUltima():
    def __init__(self, sam_checkpoint=DEFAULT_SAM_CHECKPOINT, model_type=DEFAULT_MODEL_TYPE, resnet_50_model = DEFAULT_RESNET_MODEL, fcos_model = DEFAULT_FCOS_MODEL, classes = DEFAULT_CLASSES, device = 'cpu', *args, **kwargs):
        classes = ClassLoader(classes)
        self.object_detector = FCOS(classes, torch.load(resnet_50_model, map_location=torch.device(device=device)))
        self.object_detector.load_state_dict(torch.load(fcos_model, map_location=torch.device(device=device)))
        self.object_detector.to(device=device)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        # JANK AHEAD
        self.yolo = YOLO("yolov8n.pt")
        # END JANK

        self.device = device

        self.sam_predictor = SamPredictor(self.sam)
    
    def predict(self, img, *args, **kwargs):
        self.sam_predictor.set_image(img)

        row, col = img.shape[:2]

        mcvities = preprocessing(torch.from_numpy(np.transpose(img, (2, 0, 1)))).unsqueeze(0).to(device=self.device)
        confs, locs, centers = self.object_detector(mcvities)
        boxes = fcos_to_boxes(self.object_detector.names, confs, locs, centers, row, col)
        boxes = np.array(boxes)

        if len(boxes) == 0:
            # No boxes found; return empty list
            return []
        
        r_boxes = boxes[:, 2:6] * np.array([col // 480, row // 360, col // 480, row // 360]).reshape(1, -1)

        # JANK AHEAD
        result_boxes = self.yolo(img, classes=[0])[0].boxes

        r_boxes = np.concatenate((r_boxes, result_boxes.xyxy.cpu().numpy()), axis=0)
        # JANK END

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(torch.tensor(r_boxes).to(device=self.device), img.shape[:2])

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return [*zip(boxes[:,0] + np.ones(boxes[:,0].shape), (["person", *self.object_detector.names][int(x) + 1] for x in boxes[:,0]), (x.item() for x in boxes[:,1]), masks, r_boxes), *zip((x.item() for x in result_boxes.cls), (self.yolo.names[x.item()] for x in result_boxes.cls), (x.item() for x in result_boxes.conf), masks, result_boxes.xyxy)]
    
    @staticmethod
    def main(image, *args, **kwargs):
        lu = LinnaeusUltima(*args, **kwargs)

        if isinstance(image, str):
            image = cv2.imread(image)

        results = lu.predict(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        frame = image.copy()
        for cls, clsname, conf, mask, xyxy in results:
            h, w = mask.shape[-2:]
            mask_binary = mask.reshape(h, w).cpu().numpy() * np.array([1]).reshape(1, 1)
            
            moments = cv2.moments(mask_binary, binaryImage=True)
            xcentroid = int(moments["m10"] / moments["m00"])
            ycentroid = int(moments["m01"] / moments["m00"])
            
            xmin, ymin, xmax, ymax = (int(a.item()) for a in xyxy)

            color = np.array([30, 144, 255])
            mask_image = (mask.reshape(h, w, 1).cpu().numpy() * color.reshape(1, 1, -1)).astype(np.uint8)

            # draw stuff
            if frame is not None:
                frame = cv2.addWeighted(frame, 1, mask_image, 0.6, 0)
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
                frame = cv2.putText(frame, f"{clsname[:3]} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (255, 40, 0), 1)
                frame = cv2.putText(frame, f"y={ycentroid}", (xmin, ymin + 50), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (80, 0, 200), 1)
        cv2.imshow(f'WOW!', cv2.resize(frame, (1920, 1080)))
        cv2.waitKey(0) & 0xFF == ord('q')




