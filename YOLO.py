import os
import random
import requests
import torch 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrForObjectDetection
from ultralytics import YOLO, RTDETR
import cv2

model = YOLO('D:/.pt')

results = model. predict(source="0", show=True)

print(results)
