#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import os
import supervision as sv
import cv2
from PIL import Image
import supervision as sv
import groundingdino.datasets.transforms as T
import torch
from torchvision.ops import box_convert


# In[14]:


import os

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
if os.path.isfile(WEIGHTS_NAME) is False:
    get_ipython().system(
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
if (
    os.path.isfile("GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py")
    is False
):
    get_ipython().system("git clone https://github.com/IDEA-Research/GroundingDINO.git")


# In[15]:


os.path.isfile("GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py")


# In[16]:


from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(
    "GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth",
)


# In[17]:
IMAGE_FOLDER = "output"
LICENSE_PLATE_FOLDER = "License Plate"
TEXT_PROMPT = "Number Plate"
BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.3
for file_name in os.listdir(IMAGE_FOLDER):
    if file_name.endswith(".jpg"):
        # load image
        image_path = os.path.join(IMAGE_FOLDER, file_name)
        image_source, image = load_image(image_path)
        # detect license plates
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
        if len(boxes) == 0:
            continue
        # save license plate images to separate folder
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)
        image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            license_plate_file_name = f"{os.path.splitext(file_name)[0]}_{i}.jpg"
            license_plate_file_path = os.path.join(
                LICENSE_PLATE_FOLDER, license_plate_file_name
            )
            cv2.imwrite(license_plate_file_path, image_source[y1:y2, x1:x2])


# %%
