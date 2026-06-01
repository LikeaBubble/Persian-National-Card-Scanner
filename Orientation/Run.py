import torch
import numpy as np 
import cv2
from ultralytics import YOLO
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoOrient:
    def __init__(self):
        path = Path(__file__).parent
        self.model = YOLO(f'{path}/Weights/best3.pt')

    
    def rotate(self,orig):
        with torch.no_grad():
            res = self.model.predict(orig,device=device)
        corners = res[0].keypoints.xy.cpu().numpy()
        rotated = self.deskew_card(orig,corners)
        return rotated
    
    def deskew_card(self, image, corners):
        TARGET_WIDTH = 1080
        TARGET_HEIGHT = int(TARGET_WIDTH / 1.574)
        
        destination_points = np.array([
            [0, 0],                                # Top-Left
            [TARGET_WIDTH - 1, 0],                 # Top-Right
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], # Bottom-Right
            [0, TARGET_HEIGHT - 1]                 # Bottom-Left
        ], dtype="float32")
        
        source_points = np.array(corners, dtype="float32")
        
        transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        
        flattened_image = cv2.warpPerspective(
            image, 
            transformation_matrix, 
            (TARGET_WIDTH, TARGET_HEIGHT),
            flags=cv2.INTER_LINEAR
        )
        
        return flattened_image