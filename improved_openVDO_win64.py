# Modified from https://github.com/elmonkey/Python_OpenNI2
# require-package primesense opencv-python-contrib
import time
import numpy as np
import cv2 as cv

ENABLE_CAM_COLOR = True
ENABLE_CAM_DEPTH = True
ENABLE_CAM_IR = True

# Define bits of DEPTH / IR
BITS_DEPTH = 16     # 16 or 8
BITS_IR = 16        # 16 or 8

# Define Multiply Factor of 16Bits DEPTH / IR
Depth16_MultiplyFactor = 10
IR16_MultiplyFactor = 50

# Setting VDO Path
vdo_path = './vdo/'
vdo_dir_name = 'newvdo'
vdo_dir_name = str(input("Enter VDO_DIR_NAME <= "))
vdo_path += vdo_dir_name + '/'
color_stream = cv.VideoCapture(vdo_path+"color.avi")
depth_stream = cv.VideoCapture(vdo_path+"depth"+str(BITS_DEPTH)+".avi")
ir_stream = cv.VideoCapture(vdo_path+"ir"+str(BITS_IR)+".avi")

if not color_stream.isOpened():
    print("-> Cannot found/open color vdo !!")
    ENABLE_CAM_COLOR = False
    #exit()
if not depth_stream.isOpened():
    print("-> Cannot found/open depth vdo !!")
    ENABLE_CAM_DEPTH = False
    #exit()
if not ir_stream.isOpened():
    print("-> Cannot found/open ir vdo !!")
    ENABLE_CAM_IR = False
    #exit()

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# setting fps
set_fps = depth_stream.get(cv.CAP_PROP_FPS)
set_fps = int(set_fps)
if(set_fps<=0): # if cannot load valid value -> will be fixed
    set_fps = 30


print("started vdo streaming...")

while(cv.waitKey( int(1000 / set_fps))!=27):
    if(ENABLE_CAM_COLOR):
        ret_color,color_img = color_stream.read()
    if(ENABLE_CAM_DEPTH):
        ret_depth,depth_img = depth_stream.read()
        if BITS_DEPTH==16:
           # --- DECODE DEPTH --- 8bits x 2ch -> to 16bits 1 ch
            depth_ch1, depth_ch2, _  = cv.split(depth_img) # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
            decoded_depth = np.left_shift(np.uint16(depth_ch1.copy()),8) # 8upperbits data -> convert uint16 and shift left << 8
            decoded_depth = np.bitwise_or(decoded_depth,np.uint16(depth_ch2.copy())) # bitwise or with 8lowerbits
            decoded_depth *= Depth16_MultiplyFactor
        else:
            decoded_depth = depth_img
    if(ENABLE_CAM_IR):
        ret_ir,ir_img = ir_stream.read()
        if BITS_IR==16:
            # --- DECODE IR --- 8bits x 2ch -> to 16bits 1 ch
            ir_ch1, ir_ch2, _  = cv.split(ir_img) # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
            decoded_ir = np.left_shift(np.uint16(ir_ch1.copy()),8) # 8upperbits data -> convert uint16 and shift left << 8
            decoded_ir = np.bitwise_or(decoded_ir,np.uint16(ir_ch2.copy())) # bitwise or with 8lowerbits
            decoded_ir *= IR16_MultiplyFactor
        else:
            decoded_ir = ir_img

    # fps calucation and displaying
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    if(ENABLE_CAM_COLOR): # show fps
        cv.rectangle(color_img, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(color_img, "fps="+str(fps), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # Show Images using Opencv
    if(ENABLE_CAM_COLOR):
        cv.imshow("color_img",color_img)
    if(ENABLE_CAM_DEPTH):
        cv.imshow("decoded_depth", decoded_depth)
    if(ENABLE_CAM_IR):
        cv.imshow("decoded_ir", decoded_ir)

cv.destroyAllWindows()