# Orbbec-PythonOpenCV
Orbbec Python OpenCV OpenNI Primesense Record Color Depth IR 8/16bits
## Step 1
### Install Driver in directory "Driver-installer"
#### Or manually download at https://orbbec3d.com/develop/ in Section Download Orbbec Camera Driver for Windows/Linux
#### Or get from archive at Sep2021 at https://drive.google.com/drive/folders/1Vaj6_xWRGHD594UU1W69qljjxJhPgRRZ?usp=sharing
#### For Linux you have to install driver from SDK (which matched with your Linux Distro)
## Step 2
### Ensure you have python version 3.5 upward
### Install python packages: opencv-python, numpy, primesense
#### example by using pip: pip install opencv-python,numpy,primesense
### Install VDO Codec HFYU
## Step 3
### Run streaming code using selectmode_win64.py
### OR Record VDO to files with recordDepthVDO.py AND Play with openDepthVDO.py
###  ***** 
###  Note:  if you use Linux you have to change OpenNI redistribution in every code 
###         at the line -> openni2.initialize("./Redist") to yours ex. openni2.initialize("./Redist_linux/arm64")
###         Now (Oct2021) Official Orbbec make OpenNI redistribution inclued: arm arm64 and ubuntu16.04/18.04(x86_64)
###         Please see in ./Redist_linux

# # Modified from https://github.com/elmonkey/Python_OpenNI2