import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sep import getcrop
classes={
    13:'0',
    30:'1',
    12:'2',
    24:'3',
    32:'4',
    28:'5',
    1:'6',
    17:'7',
    35:'8',
    29:'9',
    7:'A',
    3:'B',
    15:'C',
    20:'D',
    8:'E',
    23:'F',
    33:'G',
    14:'H',
    10:'I',
    0:'J',
    6:'K',
    5:'L',
    19:'M',
    16:'N',
    26:'O',
    18:'P',
    21:'Q',
    34:'R',
    4:'S',
    22:'T',
    11:'U',
    2:'V',
    27:'W',
    31:'X',
    25:'Y',
    9:'Z'
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.vgg16(pretrained=False)
num_classes = 36 
model.classifier[6] = torch.nn.Linear(4096, num_classes)

model.load_state_dict(torch.load('vgg16_weights.pth'))



model.eval()
image_path = r'Data\data2\testing_data\A\41302.png'

image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0) 
print(image)
with torch.no_grad():
    outputs = model(image)

_, predicted = torch.max(outputs, 1)
predicted_class_index = predicted.item()
predicted_class = classes[predicted_class_index]
print('Predicted class:', predicted_class)
