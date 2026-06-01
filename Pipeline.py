import Detection.Run
import Orientation.Run
import Recognition.OCR
import Recognition.Inspection
import cv2
import numpy as np 

class pipeline:
    def __init__(self):
        self.detector = Detection.Run.Detector()
        self.auto_orient = Orientation.Run.AutoOrient() 
        self.ocr = Recognition.OCR.hezar_ocr()
        self.validator = Recognition.Inspection.inspection()
        
    def execute(self,img_bytes):
        
        image = np.frombuffer(img_bytes, np.uint8)
        orig = cv2.imdecode(image, cv2.IMREAD_COLOR)
        try:
            rotated = self.auto_orient.rotate(orig)
        except:
            rotated =  orig
        
        rois = self.detector.detect(rotated)
        recognized = self.ocr.recognize(rois)
        out = self.validator.validate(recognized)
        return out
    
    def apply_orientation(self,orig):
        rotated = self.auto_orient.rotate(orig)
        return rotated
    
    def apply_detection(self,img):
        rois = self.detector.detect(img)
        return rois
    
    def apply_recognition(self,rois):
        recognized = self.ocr.recognize(rois)
        out = self.validator.validate(recognized)
        return out
        
