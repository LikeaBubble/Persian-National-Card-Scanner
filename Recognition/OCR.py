from hezar.models import Model
from pathlib import Path
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class hezar_ocr:
    def __init__(self):
        path = Path(__file__).parent
        model_dir = path / "hezar_cache/"
        try:
            self.ocr = Model.load(
                model_dir,
                load_locally=True,
                load_preprocessor=True,
                model_filename='model.pt',
                config_filename='model_config.yaml')
        except:
            return None
    
    
    def recognize(self,rois):
        output = {}
        for k,v in rois.items():
            if v is None:
                output[k] = 'Unknown'
                continue
            img = self.preprocess(v)
            texts = self.ocr.predict(img,device)
            word = ''
            for txt in texts:
                word += txt['text']
            output[k] = word
        return output 
    
    def preprocess(self,image):
        scale_factor = 3
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)
        
        upscaled = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        return upscaled