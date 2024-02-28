# Semester thesis real life drone repository of Alberto Schiaffino, adapted from Konstantin Kalenberg

## Patches to be applied to the crazyflie firmware
- Avoid height jumps when flying over obstacles: Replace `crazyflie-firmware/src/modules/src/kalman_core/mm_tof.c` with the file content of `patches/mm_tof.c` 
- Enable UART1 DMA: Enable `#define ENABLE_UART1_DMA` flag in `crazyflie-firmware/src/drivers/src/uart1.c`

---
## Fully onboard control: Autonomous crazyflie
- This part of the code provides the code for the CrazyFlie STM32 and the CrazyFlie AI-Deck to run the models fully on-board the drone itself. It provides the full  firmware for flying the drone.

### Build and flash procedure
#### STM32
- Move to `crazyflie_code/crazyflie_app`
- Run `make all` to build the custom app for the STM32 
- Put the crazyflie into bootloader mode and run `make cload` to flash code onto crazyflie. Now this app will run every time once the drone is restarted

#### Ai deck
- Move to `crazyflie_code/aideck_app`
- Connect the Ai deck via JTAG and OLIMEX to your computer
- Restart the drone after you flashed the STM32, else the next step will not work
- Source `~/gap_sdk/configs/ai_deck.sh` and run `make all flash`. You need to repeat this step every time you want to build the AI-Deck code as there is a bug in the GAP SDK. This will flash the Ai deck code onto the AI-Deck and run on every restart of the drone.

### Flying
- Restart the drone and set it down
- Run `cfclient` to start the crazyflie client. Move to the `Parameters` tab and set the parameter `PARAM/start_flying` to 1. This will make the crazyflie take-off and autonomously navigate using the pipeline developped in my master thesis. To land, set the parameter `PARAM/landing` to 1.


#### Change height 
- The project at the moment runs at 1.0m height. If you would like to change that, please change the PULP_TARGET_H variable in the "crazyflie_code/crazyflie_app/include/config_values.h" and the desiredFlightHeight in the "patches/mm_tof.c" (remember to move the file again inside your crazyflie firmware).
