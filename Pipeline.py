import json
import Detection.Run
import Orientation.Run
import Recognition.OCR
import Recognition.Inspection



class pipeline:
    """
        Its the main module which gathers all processes in one.
        Gets an image and return the result(dict).
        Detector and AutoOrient modules can use different models; currently following models 
        are available:
        Detectors:
            - yolov8 nano
            - yolov8 small
            - yolov8 medium
        
        Classifiers:
            - Resnet50
            - Efficientnet B4
            - Mobilenet Large
    """
    def __init__(self,detector,cls_model='mobilenet_large'):
        self.detector = Detection.Run.Detector(detector)
        self.auto_orient = Orientation.Run.AutoOrient(cls_model) 
        self.ocr = Recognition.OCR.hezar_ocr()
        self.validator = Recognition.Inspection.inspection()
        
    def execute(self,img):
        angle,rotated = self.auto_orient.rotate(img)
        rois = self.detector.detect(rotated)
        recognized = self.ocr.recognize(rois)
        out = self.validator.validate(recognized)
        return out,angle
