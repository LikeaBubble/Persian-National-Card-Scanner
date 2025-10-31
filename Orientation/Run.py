import torch.nn as nn
import torchvision.transforms as T
import torch
from pathlib import Path
from PIL import Image
from torchvision.models import resnet50
from torchvision.models import efficientnet_b4
from torchvision.models import mobilenet_v3_large


i2c = {0: '0', 1: '135', 2: '180', 3: '224', 4: '270', 5: '315', 6: '45', 7: '90'}
classes_num = len(i2c)
imgsz = 640
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoOrient:
    def __init__(self,model='resnet50'):
        path = Path(__file__).parent
        self.mobilenet_weight_path = path / 'Weights' / 'mobilenetlarge_78.pth'
        self.resnet50_weight_path = path / 'Weights' / 'resnet50_86.pth'
        self.efficientnet_b4_weight_path = path / 'Weights' / 'efficientnetb4_83.pth'
        
        self.model = ''
        self.current_model = None
        self.set_model(model)
        self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(imgsz),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def get_resnet50(self):
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features,classes_num)
        state_dict = torch.load(self.resnet50_weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)



    def get_efficientnet_b4(self):
        model = efficientnet_b4()
        model.classifier[1] = nn.Linear(1792,classes_num)
        state_dict = torch.load(self.efficientnet_b4_weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)


    def get_mobilenet_larg(self):
        model = mobilenet_v3_large()
        model.classifier[3] = nn.Linear(1280,classes_num)
        state_dict = torch.load(self.mobilenet_weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
    
    def rotate(self,orig):
        transformed = self.transform(orig).unsqueeze(0).to(device)
        res = self.current_model(transformed)
        
        top_id = torch.argmax(res,1).item()
        angle = int(i2c[top_id])
        
        rotated = orig.rotate(-angle)
        
        return angle,rotated
        
    
    def set_model(self,name):
        if name != self.model:
            self.model = name
            if name == 'resnet50':
                self.current_model = self.get_resnet50() 
                return 'Model successfully changed to Resnet50!'
            if name == 'mobilenet_large':
                self.current_model = self.get_mobilenet_larg()
                return 'Model successfully changed to Mobilenet_large!'
            if name == 'efficientnet_b4':
                self.current_model = self.get_efficientnet_b4()
                return 'Model successfully changed to Efficientnet_b4!'
            else:
                return 'Invalid model name! Available models : resnet50, mobilenet_large, efficientnet_b4'
            
        return f'The {name} is already in use!'