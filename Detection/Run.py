import torch
from pathlib import Path
from ultralytics import YOLO 



i2c = {0: 'birth_date', 1: 'expire_date', 2: 'father_name', 3: 'first_name', 4: 'id', 5: 'last_name'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Detector:
    def __init__(self,conf=0.25 ,iou=0.7 ,max_det=6):
        model_dir = Path(__file__).parent
        self.model = YOLO(f'{model_dir}/Weights/best2.pt',task='detect')
        self.conf = conf 
        self.max_det = max_det
        self.iou = iou
    
    def detect(self,img):
        
        with torch.no_grad():
            res = self.model.predict(
                img,
                conf= self.conf,
                iou= self.iou,
                max_det = self.max_det,
                device=device
                )
        
        image,top_rois = self.top_per_class(res[0])
        result = self.crop(image,top_rois)
        
        return result
    
    # Gets the result of detection and chooses the top confidences per each class (only one detection per class remains)
    # Then returns related bounding boxes in xyxy format and also original image
    # If there was not any prediction for a class it will return None value in dictionary for the class 
    def top_per_class(self,result):
        
        image = result.orig_img
        classes = result.boxes.cls
        confs = result.boxes.conf
        boxes = result.boxes.xyxy

        sorted_idxs = torch.sort(confs,descending=False)[1]

        confs = confs[sorted_idxs]
        boxes = boxes[sorted_idxs]
        classes = classes[sorted_idxs]

        top_rois = {'birth_date': None,
                    'expire_date': None,
                    'father_name': None,
                    'first_name': None,
                    'id': None,
                    'last_name': None}
        
        for cls,box,conf in zip(classes,boxes,confs):
            top_rois[i2c[cls.item()]] = box
        
        return image,top_rois
    
    # Gets the original image and RoI values and the returns the cropped RoIs
    def crop(self,image,top_rois):
        cropped = {}
    
        for cls,box in top_rois.items():
            if box is None:
                cropped[cls] = None
            else:
                x1,y1,x2,y2 = map(int,box)
                cropped[cls] = image[y1:y2,x1:x2]
        
        return cropped