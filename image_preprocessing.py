import os
import cv2
import numpy as np

datasets = ['gonesse','rothschild', 'google', 'noor']

def equalize_clahe_image(image_path):
    img = cv2.imread(image_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    cv2.imwrite(image_path, cl)

for dataset in datasets:
    image_folder = os.path.join('data', dataset, 'JPEGImages')

    for filename in os.listdir(image_folder):
        print(os.path.join(image_folder, filename))
        equalize_clahe_image(os.path.join(image_folder, filename))
