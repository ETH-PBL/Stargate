# Building, flashing and starting the drone code 

## Prerequisites
- Set up the crazyflie-firmware as described [here](https://github.com/bitcraze/crazyflie-firmware/blob/master/docs/building-and-flashing/build.md)
  The newest crazyflie-firmware commit we tested functionality with is: f61da11d54b6d54c7fa746688e8e9be4edd73a29
- Install the cfclient as described [here](https://github.com/bitcraze/crazyflie-clients-python/blob/master/docs/installation/install.md)
  
## Patches to be applied to the Crazyflie firmware
- Avoid height jumps when flying over obstacles: Replace `crazyflie-firmware/src/modules/src/kalman_core/mm_tof.c` with the file content of `patches/mm_tof.c` 
- Enable UART1 DMA: Enable `#define ENABLE_UART1_DMA` flag in `crazyflie-firmware/src/drivers/src/uart1.c`

## Build and flash procedure
#### STM32
- Move to `crazyflie_code/crazyflie_app`
- Run `make all` to build the custom app for the STM32 
- Put the Crazyflie into bootloader mode (press the button long) and run `make cload` to flash code onto the Crazyflie. Now, this app will run every time the drone is restarted

#### AI deck
- Move to `crazyflie_code/aideck_app`
- Every time you execute `make all` after `make clean` (and the first time you compile), you need to do a slight modification to use the same L1 buffer for both networks (in the code that is generated):
  comment out `gate_classifier_model_v51_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 49196);` and `if (gate_classifier_model_v51_L1_Memory == 0) return 4;` in `aideck_app/BUILD_MODEL/gate_classifier_model_v51Kernels.c` on lines 1394/1395.
- Connect the AI deck via JTAG and OLIMEX to your computer
- Restart the drone after you flashed the STM32, else the next step will not work
- Source `~/gap_sdk/configs/ai_deck.sh` and run `make all flash`. You need to repeat this step every time you want to build the AI Deck code. This will flash the AI deck code onto the AI deck and run on every restart of the drone.

## Flying
- Restart the drone and set it down
- Run `cfclient` to start the Crazyflie client. Move to the `Parameters` tab and set the parameter `PARAM/start_flying` to 1. This will make the Crazyflie take-off and autonomously navigate using Stargate. To land, set the parameter `PARAM/landing` to 1.


#### Change height 
- The project at the moment runs at 1.0m height. If you would like to change that, please change the PULP_TARGET_H variable in the "crazyflie_code/crazyflie_app/include/config_values.h" and the desiredFlightHeight in the "patches/mm_tof.c" (remember to move the file again inside your crazyflie firmware).
