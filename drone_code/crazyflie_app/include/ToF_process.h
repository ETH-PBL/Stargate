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
// Iman Ostovar 

#ifndef TOF_PROCESS // include guard
#define TOF_PROCESS

#include <stdint.h>
#include <stdbool.h>
#include "vl53l5cx_api.h"

//Parametrs
#define GROUND_BORDER 6
#define CELLING_BORDER 2
#define MIN_PIXEL_NUMBER 1
// Decision Parameters
#define DIS_GROUND_MIN 10 // mm
#define DIS_CEILING_MIN 10 // mm

#define TURN_MAX 1.0f
#define TURN_SLOW 0.35f
#define TURN_FAST 1.0f

#define TURN_NOT 0.0f
#define VEL_STOP 0.0f
#define VEL_FEAR -0.5f 
#define VEL_SCALE_MEDIUM 0.6f 
#define VEL_SCALE_SLOW 0.4f 
#define VEL_UP 0.3f 
#define VEL_DOWN -0.6f 
#define MAX_TURN_RATIO 0.8f
#define EPSILON 0.0001f

// #define IMAGE_PROCESS_DEBUG_PRINT

// Constants
#define ROW 8
#define COL 8
#define MAX_TARGET_NUM 6
//ToF
#define ToF_DISTANCES_LEN             (2*ROW*COL)
#define ToF_TARGETS_DETECTED_LEN      (ROW*COL)
#define ToF_TARGETS_STATUS_LEN        (ROW*COL)
#define TOF_FPS        (15.0f)
#define TOF_PERIOD        (1.0f/TOF_FPS)

// Process Parameter
#define MAX_DISTANCE_TO_PROCESS       (2000)  //mm


//structers

typedef struct pos {
	float x;
	float y;
}pos_t;


typedef struct zone {
	pos_t top_left;
	pos_t bottom_righ;
}zone_t;

typedef struct borders {
	int8_t top;
	int8_t left;
	int8_t right;
	int8_t bottom;
}borders_t;

typedef struct target {
	pos_t position;
    borders_t borders;
	uint16_t min_distance;
	uint16_t max_distance;
	uint16_t avg_distance;
    uint8_t pixels_number;
}target_t;

enum Region{RegionError = -1,
		    UpperLeft=0,
			UpperRight, 
			LowerLeft, 
			LowerRight};

enum MoveCommand{
				 CommandError=-1,
				 Stop = 0,
				 Fast_Right,
    			 Right,
				 care_forward_and_slow_right,
				 slow_forward_and_right,
				 Fast_Left,
				 left,
				 care_forward_and_slow_left,
				 slow_forward_and_left,
				 increase_altitude,
				 decrease_altitude,
				 care_forward,
				 slow_forward,
				 forward,
				 take_off,
				 land
				 };

typedef struct _FlyCommand {
	float command_velocity_x;
	float command_velocity_z;
	float command_turn; 
}FlyCommand_t;


bool isSafe(bool M[][COL], uint8_t row, uint8_t col, bool visited[][COL]);
void DFS(bool M[][COL], uint8_t row, uint8_t col, bool visited[][COL]);
uint8_t countIslands(bool M[][COL], bool objects[][ROW][COL]);
FlyCommand_t Process_ToF_Image(VL53L5CX_ResultsData* p_tof_results);
FlyCommand_t Decision_Making(target_t* targets, uint8_t object_num);
float Handle_Exception_Commands(float current_command);


#endif /* TOF_PROCESS */