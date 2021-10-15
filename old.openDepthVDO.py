# Modified from https://github.com/elmonkey/Python_OpenNI2
# require-package primesense opencv-python-contrib
import numpy as np
import cv2 as cv
import time

# Setting VDO Path
vdo_path = './vdo/'
vdo_dir_name = 'newvdo'
vdo_dir_name = str(input("Enter VDO_DIR_NAME <= "))
vdo_path += vdo_dir_name + '/'
color_stream = cv.VideoCapture(vdo_path+"color.avi")
depth_stream = cv.VideoCapture(vdo_path+"depth.avi")
ir_stream = cv.VideoCapture(vdo_path+"ir.avi")

if not color_stream.isOpened():
    print("-> Cannot open color vdo !!")
    exit()
if not depth_stream.isOpened():
    print("-> Cannot open depth vdo !!")
    exit()
if not ir_stream.isOpened():
    print("-> Cannot open ir vdo !!")
    exit()

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# setting
set_fps = depth_stream.get(cv.CAP_PROP_FPS)
set_fps = int(set_fps)
if(set_fps<=0): # if cannot load valid value -> will be fixed
    set_fps = 30

while(cv.waitKey( int(1000 / set_fps))!=27):
    ret_color,color_img = color_stream.read()
    ret_depth,depth_img = depth_stream.read()
    ret_ir,ir_img = ir_stream.read()

    if(not ret_color or not ret_depth or not ret_ir): # if error / or end of vdo
        print('Error occured or End of VDO')
        cv.waitKey(3000)
        break

    # --- DECODE DEPTH --- 8bits x 2ch -> to 16bits 1 ch
    depth_ch1, depth_ch2, _  = cv.split(depth_img) # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    decoded_depth = np.left_shift(np.uint16(depth_ch1.copy()),8) # 8upperbits data -> convert uint16 and shift left << 8
    decoded_depth = np.bitwise_or(decoded_depth,np.uint16(depth_ch2.copy())) # bitwise or with 8lowerbits

    # --- DECODE DEPTH --- 8bits x 2ch -> to 16bits 1 ch
    ir_ch1, ir_ch2, _  = cv.split(ir_img) # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    decoded_ir = np.left_shift(np.uint16(ir_ch1.copy()),8) # 8upperbits data -> convert uint16 and shift left << 8
    decoded_ir = np.bitwise_or(decoded_ir,np.uint16(ir_ch2.copy())) # bitwise or with 8lowerbits
    
    # fps calucation and displaying
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv.putText(color_img, str(fps)+"fps", (7, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 0), 3, cv.LINE_AA)
    
    # Adjusted for display
    decoded_depth*=10
    decoded_ir*=50

    cv.imshow("Color", color_img)
    cv.imshow("Depth", decoded_depth)
    cv.imshow("IR", decoded_ir)



####### for debug depth value
#print(f"decoded_depth[200,200]={decoded_depth[200,200]}")
