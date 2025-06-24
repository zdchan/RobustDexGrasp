# python library

```Shell
pip install empy catkin_pkg scipy scikit-learn cv2 opencv-python pyrealsense2 open3d numpy==1.23.1
```
    
# system library

```Shell
sudo apt install ros-noetic-hpp-fcl 
export LIB_PKGS_PTH=<your_own_path>
```
    
## install casadi

```Shell
cd ${LIB_PKGS_PTH}
git clone git@github.com:casadi/casadi.git
cd casadi && mkdir build && cd build && cmake .. && make -j && sudo make install
```

## install pinocchio
```Shell
cd ${LIB_PKGS_PTH}
git clone --recursive https://github.com/stack-of-tasks/pinocchio
mkdir pinocchio/build && mkdir pinocchio/install && cd pinocchio/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_WITH_LIBPYTHON=OFF -DCMAKE_INSTALL_PREFIX=../install && make -j && make install
```

## install rtde
```Shell
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt-get update
sudo apt install librtde librtde-dev
```

## install allegro
```Shell
sudo apt install libpopt-dev libxmlrpcpp-dev ros-noetic-libcan librospack-dev librosconsole-dev

cd ${LIB_PKGS_PTH}
tar -xzvf peak-linux-driver-x.x.tar.gz
cd peak-linux-driver-x.x && make NET=NO && sudo make install && sudo modprobe pcan

cd ${LIB_PKGS_PTH}
tar -xzvf PCAN_Basic_Linux-x.x.x.tar.gz
cd PCAN_Basic_Linux-x.x.x/pcanbasic && make && sudo make install

cd ${LIB_PKGS_PTH}
git clone git@github.com:Wonikrobotics-git/allegro_hand_linux_v4.git
cd allegro_hand_linux_v4/LibBHand_64 && make && sudo make install
```

# compile
change the cmake flag in [this folder](./raisimGymTorch/CMakeLists.txt).  to build the hardware layer

set(BUILD_UR5_REAL          ON)

set(BUILD_ALLEGRO_REAL      ON)

set(BUILD_PINOCCHIO         ON)

# evaluate
run the code of real world evaluation
```Shell
cd raisimGymTorch 
python raisimGymTorch/env/envs/allegro_real/real.py
```

