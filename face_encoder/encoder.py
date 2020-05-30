import os
import torch
import torchvision as tv
import numpy as np
from .resnet import ResNet50

FACE_IMAGE_SIZE = 112
VECTOR_SIZE = 512
FACE_ENCODER_DIR = os.path.join("face_encoder", "saved_models", "ArcFace-ResNet50.pt")
FACE_ENCODER_MODEL = ResNet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_transforms = tv.transforms.Compose([
    tv.transforms.ToPILImage(), 
    tv.transforms.Resize(FACE_IMAGE_SIZE), 
    tv.transforms.ToTensor(), 
    tv.transforms.Normalize(mean=[127.5]*3, std=[128.0]*3)
])

face_encoder = FACE_ENCODER_MODEL().to(device).eval()
face_encoder.load_state_dict(torch.load(FACE_ENCODER_DIR, map_location=device))

def encode_face(img):
    '''
    args:
        img: numpy array
    '''
    img = face_transforms(img)
    img = img.to(device).unsqueeze(0)
    with torch.no_grad():
        encoding = face_encoder(img)
    encoding = encoding.squeeze(0).cpu().numpy()
    return encoding
