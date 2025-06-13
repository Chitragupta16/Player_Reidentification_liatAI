import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image

class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()  # Remove classification head
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bbox):
        """
        Extract embedding from a cropped bbox [x1, y1, x2, y2] in the frame.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop)
        input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_tensor).squeeze().cpu().numpy()
        return embedding