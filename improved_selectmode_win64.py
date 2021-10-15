# Modified from https://github.com/elmonkey/Python_OpenNI2
# require-package primesense opencv-python-contrib
import numpy as np
import cv2 as cv
from primesense import openni2
from primesense import _openni2 as c_api

openni2.initialize("./Redist")     # can also accept the path of the OpenNI redistribution

# Enable / Disable each camera
'''Note: openni_color(openni_rgb_cam) and IR streams cannot run simultaneously '''
ENABLE_CAM_COLOR = True
ENABLE_CAM_DEPTH = True
ENABLE_CAM_IR = False
OpenNI_RGB_CAMERA = False # True if rgb_openni_cam ex. Astra ASTRA S or ASTRA MINI S // False if rgb_normal_cam ex. ASTRA PRO or Astra Stereo S U3

# Define bits of DEPTH / IR
BITS_DEPTH = 16     # 16 or 8
BITS_IR = 16        # 16 or 8

# Define Multiply Factor of 16Bits DEPTH / IR
Depth16_MultiplyFactor = 10
IR16_MultiplyFactor = 50

dev = openni2.Device.open_any()

if(ENABLE_CAM_COLOR):
    # ----------------------- RGB Camera INFO
    if OpenNI_RGB_CAMERA:
        # ----------------------- Color INFO
        color_sensor_info = dev.get_sensor_info(sensor_type=openni2.SENSOR_COLOR)
        color_sensor_mode = color_sensor_info.videoModes
        print("----------- List of color videoModes -----------")
        for i,value in enumerate(color_sensor_mode):
            print(f"{i} -> {value}")
        color_stream = dev.create_color_stream()
        print("Enter Color Mode(Number)  <==== ", end="")
        selectedColorMode = int(input())
        color_stream.set_video_mode(color_sensor_mode[selectedColorMode])
        color_mode = color_stream.get_video_mode()
        print("##### ------> Selected Color Mode(Number) only RBG888 :",color_mode)
        color_stream.start()
    else :
        color_stream = cv.VideoCapture(0) # if used ni camera please use -> dev.create_color_stream()
        if not color_stream.isOpened():
            print("Cannot open color camera!!")
            exit()

if(ENABLE_CAM_DEPTH):
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

if(ENABLE_CAM_IR):
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

# define stream fps
fps = 30
fetchFrameInterval = int(1000/fps) # 1000/fps

print("started streaming...")

while(cv.waitKey( fetchFrameInterval )!=27):

    if(ENABLE_CAM_COLOR):
        if OpenNI_RGB_CAMERA:
            color_frame = color_stream.read_frame()
            color_frame_data = color_frame.get_buffer_as_uint8()
            color_img = np.fromstring(color_frame_data,dtype=np.uint8).reshape(color_mode.resolutionY,color_mode.resolutionX,3)
            color_img = cv.cvtColor(color_img,cv.COLOR_RGB2BGR)
        else:
            ret, color_img = color_stream.read()
            color_img = cv.flip(color_img,1)

    if(ENABLE_CAM_DEPTH):
        depth_frame = depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        #depth_frame_data = depth_frame.get_buffer_as_uint8()
        depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16).reshape(depth_mode.resolutionY, depth_mode.resolutionX)
        if BITS_DEPTH==16:
            depth_img *= Depth16_MultiplyFactor
        else:
            depth_img = np.uint8(depth_img.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
            #depth_img = cv.cvtColor(depth_img,cv.COLOR_GRAY2BGR)
            depth_img = 255 - depth_img  #Shown unknowns in black

    if(ENABLE_CAM_IR):
        ir_frame = ir_stream.read_frame()
        ir_frame_data = ir_frame.get_buffer_as_uint16()
        ir_img = np.frombuffer(ir_frame_data, dtype=np.uint16).reshape(ir_mode.resolutionY, ir_mode.resolutionX)
        if BITS_IR==16:
            ir_img *= IR16_MultiplyFactor
        else:
            ir_img = np.ndarray((ir_frame.height, ir_frame.width),dtype=np.uint16, buffer = ir_frame_data).astype(np.float32)
            ir_img = np.uint8((ir_img/ir_img.max()) * 255)
    
    # Show Images using Opencv
    if(ENABLE_CAM_COLOR):
        cv.imshow("color_img",color_img)
    if(ENABLE_CAM_DEPTH):
        cv.imshow("depth_image", depth_img)
    if(ENABLE_CAM_IR):
        cv.imshow("ir_image", ir_img)

# Safe stop/release cameras
if(ENABLE_CAM_COLOR):
    if OpenNI_RGB_CAMERA:
        color_stream.stop()
    else:
        color_stream.release()
if(ENABLE_CAM_DEPTH):
    depth_stream.stop()
if(ENABLE_CAM_IR):
    ir_stream.stop()
openni2.unload()
cv.destroyAllWindows()