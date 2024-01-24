# Stargate: Multimodal Sensor Fusion for Autonomous Navigation on Miniaturized UAVs
**Authors: *Konstantin Kalenberg*, *Hanna Müller*, *Tommaso Polonelli*,  *Alberto Schiaffino*, *Vlad Niculescu*, *Cristian Cioflan*, *Michele Magno*, *Luca Benini*** 

Corresponding author: *Hanna Müller* <hanmuell@iis.ee.ethz.ch>  

<img style="float: right;" src="cover.png" width="100%">

## About the Project
Autonomously navigating robots need to perceive and interpret their surroundings. Currently, cameras are among the most used sensors due to their high resolution and frame rates at relatively low energy consumption and cost. In recent years, cutting-edge sensors, such as miniaturized depth cameras, have demonstrated strong potential, specifically for nano-size unmanned aerial vehicles (UAVs), where low power consumption, lightweight hardware, and low computational demand are essential. However, cameras are limited to working under good lighting conditions, while depth cameras have a limited range. To maximize robustness, we propose to fuse a millimeter form factor 64 pixel depth sensor and a low-resolution grayscale camera. In this work, a nano-UAV learns to detect and fly through a gate with a lightweight autonomous navigation system based on two tinyML convolutional neural network models trained in simulation, running entirely onboard in 7.6 ms and with an accuracy above 91%. Field tests are based on the Crazyflie 2.1, featuring a total mass of 39 g. We demonstrate the robustness and potential of our navigation policy in multiple application scenarios, with a failure probability down to 1.2 · 10-3 crash/meter, experiencing only two crashes on a cumulative flight distance of 1.7 km.

## How to run the training script
`cd training_quantization/deep_learning/` \
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 main_deep_learning.py`

## Hardware Setup
https://github.com/ETH-PBL/Matrix_ToF_Drones
