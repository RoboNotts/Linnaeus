from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2

DEFAULT_SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_YOLO_MODEL = "yolov8m.pt"

class LinnaeusUltima():
    def __init__(self, sam_checkpoint=DEFAULT_SAM_CHECKPOINT, model_type=DEFAULT_MODEL_TYPE, yolo_model = DEFAULT_YOLO_MODEL, device = 'cpu'):
        self.yolo = YOLO(yolo_model)
        self.yolo.to(device=device)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.sam_predictor = SamPredictor(self.sam)
    
    def predict(self, img):
        self.sam_predictor.set_image(img)
        results = self.yolo(img)
        result_boxes = results[0].boxes

        # TODO: This is a hack. We need to figure out how to use pytorch only
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(result_boxes.xyxy, img.shape[:2])

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return zip((x.item() for x in result_boxes.cls), (self.yolo.names[x.item()] for x in result_boxes.cls), (x.item() for x in result_boxes.conf), masks, result_boxes.xyxy)
    
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
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                
                plt.gca().imshow(mask_image)
                plt.gca().text(x, y, f"{clsname} {conf:0.2f}", color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

            plt.axis('off')
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Skipping visualization.")

            for cls, mask in results:
                print(f"{cls}: {mask}")
            return
        



