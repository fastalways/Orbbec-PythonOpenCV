# require-package primesense opencv-python-contrib
import numpy as np
import cv2


color_stream = cv2.VideoCapture("rawdepth.avi") #dev.create_color_stream()
if not color_stream.isOpened():
    print("Cannot open color camera!!")
    exit()

while(cv2.waitKey(34)!=27):
    ret,depth_img = color_stream.read()
    cv2.imshow("depth", depth_img)
    print(f"max={depth_img.max()},min={depth_img.min()}")
