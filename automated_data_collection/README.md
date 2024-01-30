# Generate Your Own Synthetic Data
The [open-source dataset](https://zenodo.org/records/10546408) provided with our work was generated using the [Webots simulator](https://cyberbotics.com/). We provide the sensor settings for the simulated CrazyFlie drone, simulated camera and simulated time-of-flight sensor in the `crazyflie_webots_sensor_settings.wbt` file.

## Installation Procedure
* Clone the [Bitcraze CrazyFlie simulation](https://github.com/bitcraze/crazyflie-simulation).
* Follow the [instructions](https://www.bitcraze.io/documentation/repository/crazyflie-simulation/main/installing/install_webots/).
* Follow the [examples](https://www.bitcraze.io/documentation/repository/crazyflie-simulation/main/user_guides/webots_keyboard_control/).
* Now you can modify the `webots/worlds/crazyflie_world.wbt` file from the Bitcraze Crazyflie simulation repository using our sensor setup from the `crazyflie_webots_sensor_settings.wbt` file.
* To implement path planning and following logic, you can get inspiration in the Bitcraze Crazyflie simulation repository `simulator_files/webots/controllers/crazyflie_controller_py/crazyflie_controller_py.py` file.

## Disclaimer
Be aware that we perform processing operations such as augmentations and cropping of the camera and time-of-flight sensor as described in the published report. The provided [open-source dataset](https://zenodo.org/records/10546408) has already been processed and thus might appear different than what the sensor settings in `crazyflie_webots_sensor_settings.wbt` produce. To reproduce the processing steps on your own synthetic data, please refer to the report.


