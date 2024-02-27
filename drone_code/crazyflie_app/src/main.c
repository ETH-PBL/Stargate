// Copyright (C) 2022-2024 ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// SPDX-License-Identifier: GPL-3.0
// ======================================================================

// Authors: 
// Konstantin Kalenberg, ETH Zurich
// Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
// Tommaso Polonelli, ETH Zurich
// Alberto Schiaffino, ETH Zurich
// Vlad Niculescu, ETH Zurich
// Cristian Cioflan, ETH Zurich
// Michele Magno, ETH Zurich
// Luca Benini, ETH Zurich

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>


#include "app.h"
#include "FreeRTOS.h"
#include "system.h"
#include "task.h"
#include "debug.h"
#include "stabilizer_types.h"
#include "estimator_kalman.h"
#include "commander.h"
#include "log.h"
#include "param.h"
#include "uart_dma_setup.h"
#include "config_values.h"
#include "uart1.h"

// ToF
#include "vl53l5cx_api.h"
#include "crtp.h"
#include "deck.h"
#include "I2C_expander.h"
#include "ToF_process.h"

#define VL53L5CX_FORWARD_I2C_ADDRESS            ((uint16_t)(VL53L5CX_DEFAULT_I2C_ADDRESS*4))

#define TOF_ROWS 8
#define TOF_COLS 8
#define TOF_DISTANCES_LEN             (2 * TOF_ROWS * TOF_COLS)
#define TOF_TARGETS_DETECTED_LEN      (TOF_ROWS * TOF_COLS)
#define TOF_TARGETS_STATUS_LEN        (TOF_ROWS * TOF_COLS)
#define TOF_MAX_RANGE 3000

#define  ARM_CM_DEMCR      (*(uint32_t *)0xE000EDFC)
#define  ARM_CM_DWT_CTRL   (*(uint32_t *)0xE0001000)
#define  ARM_CM_DWT_CYCCNT (*(uint32_t *)0xE0001004)

#define DEBUG_MODULE "IMAV_CHALLENGE_APP"
#define CNN_RESULT_BUFFERSIZE 3

static VL53L5CX_Configuration vl53l5dev_f;

// Function defines
bool initialize_sensors_I2C(VL53L5CX_Configuration *p_dev, uint8_t mode);
bool config_sensors(VL53L5CX_Configuration *p_dev, uint16_t new_i2c_address);
bool get_sensor_data(VL53L5CX_Configuration *p_dev,VL53L5CX_ResultsData *p_results);
float smoothing_velocity(float forward_velocity, float new_velocity); 

// Global vars
int8_t cnn_buffer[CNN_RESULT_BUFFERSIZE];
static float cnn_data_float[2];
static float yaw_rate_scale_cnn = 1.5f;
static uint8_t random_turn_timeout = 90;  //140
static uint8_t random_turn_length = 7; //18
volatile uint8_t dma_flag = 0;
static uint8_t start_command = 0;
static uint8_t land_command = 0;
uint64_t t0, t_frame, t_prev;
int idx_safety = 0;
int idx_navigation = 1;
int idx_classification = 2;
int8_t safety_byte_value = -128;

void decode_and_process_tof_results(VL53L5CX_ResultsData* p_tof_results, uint16_t* tof_distances_vector_processed_mm)
{
  // Decode results
  uint16_t tof_distances[TOF_DISTANCES_LEN/2];
  uint8_t tof_targets[TOF_TARGETS_DETECTED_LEN];
  uint8_t tof_status[TOF_TARGETS_STATUS_LEN];
  memcpy(tof_distances, (uint8_t *)(&p_tof_results->distance_mm[0]), TOF_DISTANCES_LEN);
  memcpy(tof_targets, (uint8_t *)(&p_tof_results->nb_target_detected[0]), TOF_TARGETS_DETECTED_LEN);
  memcpy(tof_status, (uint8_t *)(&p_tof_results->target_status[0]), TOF_TARGETS_STATUS_LEN);

  // Find invalid pixels
  bool invalid_mask[TOF_ROWS*TOF_COLS];
  for (int i = 0; i < TOF_ROWS * TOF_COLS; ++i)
  {
    invalid_mask[i] = (tof_status[i] != 5 && tof_status[i] != 9) || tof_targets[i] != 1;
  }

  // Process to use in CNN, still remains in [mm]
  for (int i = 0; i < TOF_ROWS; ++i)
  {
    for (int j = 0; j < TOF_COLS; ++j)
    {
      if(!invalid_mask[j+TOF_COLS*i] && tof_distances[j+TOF_COLS*i] <= TOF_MAX_RANGE) // check if the pixel is valid
      {
        tof_distances_vector_processed_mm[j+TOF_COLS*i] = tof_distances[j+TOF_COLS*i];
      }
      else
      {
        tof_distances_vector_processed_mm[j+TOF_COLS*i] = TOF_MAX_RANGE;
      }
    }
  }
}

void process_cnn_output(int8_t* cnn_output_8b, float* cnn_output_float)
{
  	cnn_output_float[0] = (float) cnn_output_8b[idx_navigation] * SCALE_QT_NAVIGATION_OUTPUT;
	cnn_output_float[1] = (float) cnn_output_8b[idx_classification] * SCALE_QT_CLASSIFICATION_OUTPUT;
}

static setpoint_t create_setpoint(float x_vel, float z, float yaw_rate)
{
	setpoint_t setpoint;
	memset(&setpoint, 0, sizeof(setpoint_t));
	setpoint.mode.x = modeVelocity;
	setpoint.mode.y = modeVelocity;
	setpoint.mode.z = modeAbs;
	setpoint.mode.yaw = modeVelocity;

	setpoint.velocity.x	= x_vel;
	setpoint.velocity.y	= 0.0f;
	setpoint.position.z = z;
	setpoint.attitudeRate.yaw = yaw_rate;
	setpoint.velocity_body = true;
	return setpoint;
}

void headToPosition(float x, float y, float z, float yaw)
{
	setpoint_t setpoint;
	memset(&setpoint, 0, sizeof(setpoint_t));

	setpoint.mode.x = modeAbs;
	setpoint.mode.y = modeAbs;
	setpoint.mode.z = modeAbs;
	setpoint.mode.yaw = modeAbs;

	setpoint.position.x = x;
	setpoint.position.y = y;
	setpoint.position.z = z;
	setpoint.attitude.yaw = yaw;
	commanderSetSetpoint(&setpoint, 3);
}

void takeoff_controller(float height)
{
	point_t pos;
	memset(&pos, 0, sizeof(pos));
	estimatorKalmanGetEstimatedPos(&pos);

	int endheight = (int)(100*(height-0.2f));
	for(int i=0; i<endheight; i++)
	{
		headToPosition(pos.x, pos.y, 0.2f + (float)i / 100.0f, 0);
		vTaskDelay(50);
	}

	for(int i=0; i<100; i++)
	{
		headToPosition(pos.x, pos.y, height, 0);
		vTaskDelay(50);
	}
}

void land_controller(void){
	point_t pos;
	memset(&pos, 0, sizeof(pos));
	estimatorKalmanGetEstimatedPos(&pos);

	float height = pos.z;
	for(int i=(int)100*height; i>5; i--)
	{
		headToPosition(pos.x, pos.y, (float)i / 100.0f, 0);
		vTaskDelay(20);
	}
	vTaskDelay(200);
}

void appMain()
{
	DEBUG_PRINT("IMAV CHALLENGE MODEL CF APP started! \n");

  	// Start System
	vTaskDelay(3000);
	systemWaitStart();
	vTaskDelay(1000);
	USART_DMA_Start(921600, cnn_buffer, CNN_RESULT_BUFFERSIZE);
	uart1Init(921600);

	// Initialize the ToF deck
	bool gpio_exp_status = false;
	bool sensors_status = true;
	gpio_exp_status = I2C_expander_initialize();
	DEBUG_PRINT("ToFDeck I2C_GPIO Expander: %s\n", gpio_exp_status ? "OK." : "ERROR!");
	vTaskDelay(M2T(100));
	sensors_status = initialize_sensors_I2C(&vl53l5dev_f,1); //forward
	DEBUG_PRINT("ToFDeck Forward Sensor Initialize 1: %s\n", sensors_status ? "OK." : "ERROR!");

	if(gpio_exp_status == false || sensors_status == false)
	{
		DEBUG_PRINT("ERROR LOOP_1!");
		while (1)
		{//stay in ERROR LOOP
		vTaskDelay(M2T(10000));
		}
	}
  	DEBUG_PRINT("ToFDeck GPIO & Interrupt Initialized. \n");

	// Start ToF sensor ranging
	vTaskDelay(M2T(100));
	uint8_t ranging_start_res_f = vl53l5cx_start_ranging(&vl53l5dev_f);
	DEBUG_PRINT("ToFDeck Start Sensor Forward Ranging: %s\n", (ranging_start_res_f == VL53L5CX_STATUS_OK) ? "OK." : "ERROR!");

	if(ranging_start_res_f != VL53L5CX_STATUS_OK){
		DEBUG_PRINT("ERROR LOOP_2!\n");
		while (1)
		{//stay in ERROR LOOP
		vTaskDelay(M2T(10000));
		}
	}

  	uint8_t get_data_success_f = false;

  	while(!start_command)
	{
		DEBUG_PRINT("Command start using the start_flying parameter\n");
		vTaskDelay(1000);
	}

	vTaskDelay(5000);
	estimatorKalmanInit();  // reset the estimator before taking off
	takeoff_controller(PULP_TARGET_H);

  	// Variables needed in loop
	uint16_t* tof_distances_vector_processed_mm = (uint16_t*)malloc(TOF_ROWS * TOF_COLS * sizeof(uint16_t));
	FlyCommand_t flight_command_tof_obstacle_avoid;
	uint8_t timeout_counter = 0;
	uint8_t random_turn_counter = 0;
	setpoint_t setp_imav_challenge = create_setpoint(0.0f, PULP_TARGET_H, 0.0f);
	float yaw_rate_raw, probability_of_gate;
	float forward_velocity = 0.0f;
	float yaw_rate_processed = 0.0f;
	//for smoothing
	float new_velocity = 0.0f;

	uint8_t collision_avoidance_counter=0;
	flight_command_tof_obstacle_avoid.command_velocity_x=0;
	flight_command_tof_obstacle_avoid.command_velocity_z=0;

	while(1)
	{
		// If landing commanded from the client
		if (land_command==1)
		{
			land_controller();
			break;
		}

		// Obtain new ToF frame
		VL53L5CX_ResultsData vl53l5_res_f;
		get_data_success_f = get_sensor_data(&vl53l5dev_f, &vl53l5_res_f);

		if (get_data_success_f == true)
    	{
			// Process ToF, set invalid pixels to TOF_MAX_RANGE, set values above TOF_MAX_RANGE to TOF_MAX_RANGE
			decode_and_process_tof_results(&vl53l5_res_f, tof_distances_vector_processed_mm);

			// Send processed ToF frame to AI deck
			uart1SendDataDmaBlocking(TOF_ROWS * TOF_COLS * sizeof(uint16_t), (uint8_t*)tof_distances_vector_processed_mm);

			// Compute flight commands from ToF obstacle avoidance algorithm
			flight_command_tof_obstacle_avoid = Process_ToF_Image(&vl53l5_res_f);

			// Clear Flag
			get_data_success_f = false;
    	}

		// If new UART with CNN results data is available
		if (dma_flag == 1)
		{
			// Check safety byte to check order on STM32 due to bug in uart
			if (cnn_buffer[idx_safety] != safety_byte_value) {
				DEBUG_PRINT("UART SAFETY BYTE ERROR\n");
				if (cnn_buffer[0] == safety_byte_value) {
				idx_safety = 0;
				idx_navigation = 1;
				idx_classification = 2;
				}
				else if (cnn_buffer[1] == safety_byte_value) {
				idx_safety = 1;
				idx_navigation = 2;
				idx_classification = 0;
				}
				else {
				idx_safety = 2;
				idx_navigation = 0;
				idx_classification = 1;
				}
			}
			
			timeout_counter = 0;
			dma_flag = 0;  // clear the flag

			process_cnn_output(cnn_buffer, cnn_data_float);  // get scaled data
			memset(cnn_buffer, 0, CNN_RESULT_BUFFERSIZE);  // clear the dma buffer

			// Extract steering angle and probability of collision
			yaw_rate_raw = cnn_data_float[0];
			probability_of_gate = cnn_data_float[1];

			// Make decision and scale forward velocity
			if (probability_of_gate > GATE_THRESHOLD)
			{
				new_velocity = fmaxf(0.5, flight_command_tof_obstacle_avoid.command_velocity_x);
				
				forward_velocity = smoothing_velocity(forward_velocity, new_velocity);
				forward_velocity = new_velocity;
				yaw_rate_processed = yaw_rate_raw * yaw_rate_scale_cnn;   // positive means turn left
				collision_avoidance_counter=0;
			}
			else
			{
				if(collision_avoidance_counter==0){
					new_velocity = flight_command_tof_obstacle_avoid.command_velocity_x;
					forward_velocity = smoothing_velocity(forward_velocity, new_velocity);
					yaw_rate_processed = flight_command_tof_obstacle_avoid.command_turn;  // positive means turn left
				}
				collision_avoidance_counter= (collision_avoidance_counter++)%4; 
			}

		// Convert to degrees and change sign
			yaw_rate_processed = yaw_rate_processed * 180.0f / 3.14159f;

			// Randomly halt drone and turn it in place to break counterclockwise loop inherent to random exploration algo
			if (random_turn_counter >= random_turn_timeout) {
				if (random_turn_counter >= random_turn_timeout + random_turn_length || probability_of_gate > GATE_THRESHOLD){
					random_turn_counter = 0;
					setp_imav_challenge = create_setpoint(forward_velocity, PULP_TARGET_H, yaw_rate_processed);
					DEBUG_PRINT("Resume normal flight\n");
				}
				else {
					setp_imav_challenge = create_setpoint(0.0f, PULP_TARGET_H, 50.0f);   // Turn left hard
					DEBUG_PRINT("Random turn left timeout\n");
				}

			}
			else {
				setp_imav_challenge = create_setpoint(forward_velocity, PULP_TARGET_H, yaw_rate_processed);
			}
			random_turn_counter++;

		}
		else
		{
			// If no packet received for 400ms
			timeout_counter++;
			if (timeout_counter > 10)
				DEBUG_PRINT("Navigation data timeout\n");
			vTaskDelay(40);
		}

		// Send setpoint in every loop iteration
    	commanderSetSetpoint(&setp_imav_challenge, 3);
	}
}

// UART-DMA interrupt - triggered when a new inference result is available
void __attribute__((used)) DMA1_Stream1_IRQHandler(void)
{
	t_prev = t0;
	t0 = xTaskGetTickCount();
	t_frame = t0 - t_prev;
	DMA_ClearFlag(DMA1_Stream1, UART3_RX_DMA_ALL_FLAGS);
	dma_flag = 1;
}

bool config_sensors(VL53L5CX_Configuration *p_dev, uint16_t new_i2c_address)
{
  p_dev->platform = VL53L5CX_DEFAULT_I2C_ADDRESS; // use default adress for first use

  // initialize the sensor
  uint8_t tof_res = vl53l5cx_init(p_dev);   if (tof_res != VL53L5CX_STATUS_OK) return false ;

  // Configurations
  // change i2c address
  tof_res = vl53l5cx_set_i2c_address(p_dev, new_i2c_address);if (tof_res != VL53L5CX_STATUS_OK) return false ;
  tof_res = vl53l5cx_set_resolution(p_dev, VL53L5CX_RESOLUTION_8X8);if (tof_res != VL53L5CX_STATUS_OK) return false ;
  // 15hz
  tof_res = vl53l5cx_set_ranging_frequency_hz(p_dev, 15);if (tof_res != VL53L5CX_STATUS_OK) return false ;
  tof_res = vl53l5cx_set_target_order(p_dev, VL53L5CX_TARGET_ORDER_CLOSEST);if (tof_res != VL53L5CX_STATUS_OK) return false ;
  tof_res = vl53l5cx_set_ranging_mode(p_dev, VL53L5CX_RANGING_MODE_CONTINUOUS);if (tof_res != VL53L5CX_STATUS_OK) return false ;
  // tof_res = vl53l5cx_set_ranging_mode(p_dev, VL53L5CX_RANGING_MODE_AUTONOMOUS);if (tof_res != VL53L5CX_STATUS_OK) return false ;// TODO test it

  // Check for sensor to be alive
  uint8_t isAlive;
  tof_res =vl53l5cx_is_alive(p_dev,&isAlive);if (tof_res != VL53L5CX_STATUS_OK) return false;
  if (isAlive != 1) return false;

  // All Good!
  return true;
}

bool initialize_sensors_I2C(VL53L5CX_Configuration *p_dev, uint8_t mode)
{
  bool status = false;

  if (mode == 1 && p_dev != NULL){
    //enable forward only and config
    status = I2C_expander_set_register(OUTPUT_PORT_REG_ADDRESS,LPN_FORWARD_PIN | LED_FORWARD_PIN );if (status == false)return status;
    status = config_sensors(p_dev,VL53L5CX_FORWARD_I2C_ADDRESS);if (status == false)return status;
  }
  return status;
}

bool get_sensor_data(VL53L5CX_Configuration *p_dev,VL53L5CX_ResultsData *p_results){

  // Check  for data ready I2c
  uint8_t ranging_ready = 2;
  //ranging_ready --> 0 if data is not ready, or 1 if a new data is ready.
  uint8_t status = vl53l5cx_check_data_ready(p_dev, &ranging_ready);if (status != VL53L5CX_STATUS_OK) return false;

  // 1 Get data in case it is ready
  if (ranging_ready == 1){
    status = vl53l5cx_get_ranging_data(p_dev, p_results);if (status != VL53L5CX_STATUS_OK) return false;
  }else {
    //0  data in not ready yet
    return false;
  }

  // All good then
  return true;
}


float smoothing_velocity(float forward_velocity, float new_velocity){
	float actual_velocity;
	if(new_velocity > 0 && forward_velocity > 0){
		if(new_velocity>=forward_velocity){//increase smoothly 
			actual_velocity = fmin(new_velocity, forward_velocity*1.2);
		}else{//decrease smoothly
			actual_velocity = fmax(new_velocity, forward_velocity*0.8);
		}
	}else if(new_velocity >= 0 && forward_velocity < 0){//no sudden changes of direction
		actual_velocity = 0;
	}else if(forward_velocity == 0){
		if(new_velocity > 0 ){
			actual_velocity = 0.10;
		}else{
			actual_velocity = new_velocity;
		}
	}else{
		actual_velocity = new_velocity;
	}

	return actual_velocity;
}

// BE AWARE: PARAM_GROUP + PARAM_NAME + 1 <= 26 else the crazyflie firmware crashes
PARAM_GROUP_START(PARAM)
PARAM_ADD(PARAM_UINT8, start_flying, &start_command)
PARAM_ADD(PARAM_UINT8, landing, &land_command)
PARAM_GROUP_STOP(PARAM)

LOG_GROUP_START(LOG)
LOG_ADD(LOG_FLOAT, yaw_rate, &cnn_data_float[0])
LOG_ADD(LOG_FLOAT, gate_prob, &cnn_data_float[1])
LOG_GROUP_STOP(LOG)

PARAM_GROUP_START(SETTINGS)
PARAM_ADD(PARAM_FLOAT, yaw_rate_scale, &yaw_rate_scale_cnn)
PARAM_ADD(PARAM_UINT8, rand_turn_lim, &random_turn_timeout)
PARAM_ADD(PARAM_UINT8, rand_turn_len, &random_turn_length)
PARAM_GROUP_STOP(SETTINGS)
