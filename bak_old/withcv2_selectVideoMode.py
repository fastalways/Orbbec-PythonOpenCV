# require-package primesense opencv-python-contrib
import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

openni2.initialize("./Redist")     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()

# ----------------------- Depth INFO
depth_sensor_info = dev.get_sensor_info(sensor_type=openni2.SENSOR_DEPTH)
depth_sensor_mode = depth_sensor_info.videoModes
print("----------- List of depth videoModes -----------")
for i,value in enumerate(depth_sensor_mode):
    print(f"{i} -> {value}")
depth_stream = dev.create_depth_stream()
print("Enter Depth Mode(Number) only16bits  <==== ", end="")
selectedDepthMode = int(input())
depth_stream.set_video_mode(depth_sensor_mode[selectedDepthMode])
depth_mode = depth_stream.get_video_mode()
print("##### ------> Selected Depth Mode:",depth_mode)
depth_stream.start()

# ----------------------- Infrared INFO
ir_sensor_info = dev.get_sensor_info(sensor_type=openni2.SENSOR_IR)
ir_sensor_mode = ir_sensor_info.videoModes
print("----------- List of IR videoModes -----------")
for i,value in enumerate(ir_sensor_mode):
    print(f"{i} -> {value}")
ir_stream = dev.create_ir_stream()
print("Enter IR Mode(Number) only16bits <==== ", end="")
selectedIRMode = int(input())
ir_stream.set_video_mode(ir_sensor_mode[selectedIRMode])
ir_mode = ir_stream.get_video_mode()
print("##### ------> Selected IR Mode:",ir_mode)
ir_stream.start()

# ----------------------- RGB Camera INFO

color_stream = cv2.VideoCapture(0) # if used ni camera please use -> dev.create_color_stream()
if not color_stream.isOpened():
    print("Cannot open color camera!!")
    exit()



fetchFrameInterval = int(1000/depth_mode.fps) # based on depth streaming fps 

print("started streaming...")

while(cv2.waitKey( fetchFrameInterval )!=27):

    ret, color_img = color_stream.read()
    color_img = cv2.flip(color_img,1)

    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()
    #depth_frame_data = depth_frame.get_buffer_as_uint8()

    depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img.shape = (1, depth_mode.resolutionY, depth_mode.resolutionX)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)
    depth_img *= 10

    ir_frame = ir_stream.read_frame()
    ir_frame_data = ir_frame.get_buffer_as_uint16()

    ir_img = np.frombuffer(ir_frame_data, dtype=np.uint16)
    ir_img.shape = (1, ir_mode.resolutionY, ir_mode.resolutionX)
    ir_img = np.concatenate((ir_img, ir_img, ir_img), axis=0)
    ir_img = np.swapaxes(ir_img, 0, 2)
    ir_img = np.swapaxes(ir_img, 0, 1)
    ir_img *= 50


    cv2.imshow("color_img",color_img)
    cv2.imshow("depth_image", depth_img)
    cv2.imshow("ir_image", ir_img)

color_stream.release()
depth_stream.stop()
ir_stream.stop()
openni2.unload()
cv2.destroyAllWindows()