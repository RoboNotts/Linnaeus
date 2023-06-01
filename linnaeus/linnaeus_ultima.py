from linnaeus.core.models import FCOS
from linnaeus.core.loaders import ClassLoader
from linnaeus.core.data_augmentation import preprocessing
from linnaeus.core.mAP.functions import fcos_to_boxes
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch

DEFAULT_SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_FCOS_MODEL = "fcos.pt"
DEFAULT_RESNET_MODEL = "resnet50.pt"
DEFAULT_CLASSES = "classes.txt"

class LinnaeusUltima():
    def __init__(self, sam_checkpoint=DEFAULT_SAM_CHECKPOINT, model_type=DEFAULT_MODEL_TYPE, resnet_50_model = DEFAULT_RESNET_MODEL, fcos_model = DEFAULT_FCOS_MODEL, classes = DEFAULT_CLASSES, device = 'cpu', *args, **kwargs):
        classes = ClassLoader(classes)
        self.object_detector = FCOS(classes, torch.load(resnet_50_model))
        self.object_detector.load_state_dict(torch.load(fcos_model))
        self.object_detector.to(device=device)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.device = device

        self.sam_predictor = SamPredictor(self.sam)
    
    def predict(self, img, *args, **kwargs):
        self.sam_predictor.set_image(img)

        row, col = img.shape[:2]

        mcvities = preprocessing(torch.from_numpy(np.transpose(img, (2, 0, 1)))).unsqueeze(0).to(device=self.device)
        confs, locs, centers = self.object_detector(mcvities)
        boxes = np.array(fcos_to_boxes(self.object_detector.names, confs, locs, centers, row, col))

        if len(boxes) == 0:
            # No boxes found; return empty list
            return []
        
        r_boxes = boxes[:, 2:6] * np.array([col // 480, row // 360, col // 480, row // 360]).reshape(1, -1)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(torch.tensor(r_boxes).to(device=self.device), img.shape[:2])

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return zip(boxes[:,0], (self.object_detector.names[int(x)] for x in boxes[:,0]), (x.item() for x in boxes[:,1]), masks, r_boxes)
    
    @staticmethod
    def main(image, *args, **kwargs):
        lu = LinnaeusUltima(*args, **kwargs)

        if isinstance(image, str):
            image = cv2.imread(image)

        results = lu.predict(image)

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for _, clsname, conf, mask, xyxy in results:
                x, y = xyxy[:2]
                color = np.array([30/255, 144/255, 255/255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1).cpu().numpy() * color.reshape(1, 1, -1)
                
                plt.gca().imshow(mask_image)
                plt.gca().text(x, y, f"{clsname} {conf:0.2f}", color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

            plt.axis('off')
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Skipping visualization.")

            for _, clsname, conf, mask, xyxy in results:
                print(f"{clsname}: {mask}: {xyxy}")
            return
        



