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

# setting for record
fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
initVDO_flag = False
colorRecorder = cv2.VideoWriter()
depthRecorder = cv2.VideoWriter()
irRecorder = cv2.VideoWriter()

fetchFrameInterval = int(1000/depth_mode.fps) # based on depth streaming fps 

print("started streaming...")

while(cv2.waitKey( fetchFrameInterval )!=27):
    #------ color frame processing ------
    ret, color_img = color_stream.read()
    color_img = cv2.flip(color_img,1)

    #------ depth frame processing ------
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()
    #depth_frame_data = depth_frame.get_buffer_as_uint8()
    depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img.shape = (1, depth_mode.resolutionY, depth_mode.resolutionX)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)
    depth_img *= 1
    # covert depth 16bits to 2ch x 8bits / c1 -> depth8 (8 upper bits) / c2 -> depth8 (8 lower bits) /
    cdepth_img = depth_img.copy()
    cdepth_img, _, _ = cv2.split(cdepth_img) # 3ch had the same value, thus split to 1ch
    depth_ch1 = np.uint8(np.right_shift(cdepth_img,8)) # make upper bits - using right shift and convert to uint8
    depth_ch2 = np.uint8(cdepth_img) # make lower bits - using convert to uint8 which 8 upper bits will lost
    encoded_depth = cv2.merge([depth_ch1,depth_ch2,depth_ch1])

    


    #------ ir frame processing ------
    ir_frame = ir_stream.read_frame()
    ir_frame_data = ir_frame.get_buffer_as_uint16()
    #ir_frame_data = ir_frame.get_buffer_as_uint8()
    ir_img = np.frombuffer(ir_frame_data, dtype=np.uint16)
    ir_img.shape = (1, ir_mode.resolutionY, ir_mode.resolutionX)
    ir_img = np.concatenate((ir_img, ir_img, ir_img), axis=0)
    ir_img = np.swapaxes(ir_img, 0, 2)
    ir_img = np.swapaxes(ir_img, 0, 1)
    ir_img *= 1
    # covert ir 16bits to 2ch x 8bits / c1 -> ir8 (8 upper bits) / c2 -> ir8 (8 lower bits) /
    cir_img = ir_img.copy()
    cir_img, _, _ = cv2.split(cir_img) # 3ch had the same value, thus split to 1ch
    ir_ch1 = np.uint8(np.right_shift(cir_img,8)) # make upper bits - using right shift and convert to uint8
    ir_ch2 = np.uint8(cir_img) # make lower bits - using convert to uint8 which 8 upper bits will lost
    encoded_ir = cv2.merge([ir_ch1,ir_ch2,ir_ch1])

    # init vdo header
    if(not initVDO_flag):
        colorRecorder = cv2.VideoWriter("color.avi", fourcc, depth_mode.fps, (color_img.shape[1], color_img.shape[0]))
        depthRecorder = cv2.VideoWriter("depth.avi", fourcc, depth_mode.fps, (encoded_depth.shape[1], encoded_depth.shape[0]))
        irRecorder = cv2.VideoWriter("ir.avi", fourcc, depth_mode.fps, (encoded_ir.shape[1], encoded_ir.shape[0]))
        initVDO_flag = True
    # vdo write stream
    colorRecorder.write(color_img)
    depthRecorder.write(encoded_depth)
    irRecorder.write(encoded_ir)

    cv2.imshow("color_img",color_img)
    cv2.imshow("depth_image", depth_img)
    cv2.imshow("ir_image", ir_img)

print(f"color_img.dtype={color_img.dtype} color_img.shape={color_img.shape}")
print(f"encoded_depth.dtype={encoded_depth.dtype} encoded_depth.shape={encoded_depth.shape}")
print(f"encoded_depth.dtype={encoded_ir.dtype} encoded_depth.shape={encoded_ir.shape}")

color_stream.release()
depth_stream.stop()
ir_stream.stop()
openni2.unload()
cv2.destroyAllWindows()