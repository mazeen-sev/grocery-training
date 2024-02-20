import os
from ultralytics import YOLO
import cv2

if __name__ == '__main__': 
    model = YOLO('runs/detect/train6/weights/best.pt')
    image = cv2.imread("C:/Users/Jeff\Desktop/food app/final year proj/future test images/71CcmSuTAnL.jpg")

    results = model(image)