# require-package primesense opencv-python-contrib
import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

openni2.initialize("./Redist")     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()

color_stream = cv2.VideoCapture(0) #dev.create_color_stream()
if not color_stream.isOpened():
    print("Cannot open color camera!!")
    exit()

depth_stream = dev.create_depth_stream()
depth_stream.start()

ir_stream = dev.create_ir_stream()
ir_stream.start()


while(cv2.waitKey(34)!=27):

    ret, color_img = color_stream.read()

    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()

    depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img.shape = (1, 480, 640)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)
    depth_img *= 10

    ir_frame = ir_stream.read_frame()
    ir_frame_data = ir_frame.get_buffer_as_uint16()

    ir_img = np.frombuffer(ir_frame_data, dtype=np.uint16)
    ir_img.shape = (1, 480, 640)
    ir_img = np.concatenate((ir_img, ir_img, ir_img), axis=0)
    ir_img = np.swapaxes(ir_img, 0, 2)
    ir_img = np.swapaxes(ir_img, 0, 1)
    ir_img *= 100

    cv2.imshow("color_img",color_img)
    cv2.imshow("depth_image", depth_img)
    cv2.imshow("ir_image", ir_img)
    


depth_stream.stop()
ir_stream.stop()
openni2.unload()