
from torchvision import models, transforms
import torch
import torch.nn as nn
import cv2
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------ Fire Detection ------------

# Upload Image
image = cv2.imread("/Users/alibiserikbay/Downloads/dataton/demo/29.jpg")


def get_model():
    # Returning the pretrained model - ResNET34
    model = models.resnet34(pretrained=True)

    # Freezing all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(512, 128),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(128, 1),
                             nn.Sigmoid())

    # loss_function - Binary Cross Entropy
    # optimizer - Adam
    # Recommended by https://www.mdpi.com/1424-8220/22/5/1701 for Fire Classifying
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer


@torch.no_grad()
def pred(x, model):
    # Evaluating the model without training it
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = cv2.resize(image, (224, 224))
    im = torch.tensor(im/255)
    im = im.permute(2, 0, 1)
    im = normalize(im)
    im = im.unsqueeze(0)
    prediction = model(im.float())

    # if 0<prediction<0.5 => return False
    # else True
    is_correct = (prediction > 0.5)
    return is_correct.cpu().numpy().tolist()[0][0]


model, x, y = get_model()
model.load_state_dict(torch.load('resnet34 (1).pth'))
model.eval()

res = pred(image, model)
print(res)
cv2.putText(image, str(res), (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

cv2.imwrite("output.png", image)
