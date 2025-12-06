#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision import datasets



def top5_classes_from(indices):
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Get top 5 predictions
    top5_indices = indices[0, :5].tolist()
    top5_classes = [categories[i] for i in top5_indices]
    print("Top 5 predictions:")
    for i, class_name in enumerate(top5_classes):
        print(f"{i+1}: {class_name}")
    print()


#PIL Load Images Example (Not on Colab)
preprocess_example = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

curly_img = "data/train/curly/images78.jpg"
straight_img = "data/train/straight/s8.jpg"

#Use predefined model and predict without any further training.
# Load pre-trained model
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.eval()

for img_path in [curly_img, straight_img]:
    img = Image.open(img_path)
    # Resize to target size
    img = preprocess_example(img)

    batch_t = torch.unsqueeze(img, 0)

    # Make prediction
    with torch.no_grad():
        output = model(batch_t)

    _, sorted_indices = torch.sort(output, descending=True)
    top5_classes_from(sorted_indices)