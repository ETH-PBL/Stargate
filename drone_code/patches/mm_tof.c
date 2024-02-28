/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2021 Bitcraze AB
 * 
 * Modified by Konstantin Kalenberg, 2023 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "mm_tof.h"
#include "debug.h"
#include <time.h>
#include <stdio.h>

// Custom global variables for code to avoid height jumps when passing over obstacles/gate
float desiredFlightHeight = 1.0f;
float heightDiffThreshold = 0.05f;
int waitCounterTakeoff = 0;
int waitCounterObstacle = 0;
int thresholdTakeoff = 50;
int thresholdObstacle = 300;//100
int flag=0;

void kalmanCoreUpdateWithTof(kalmanCoreData_t* this, tofMeasurement_t *tof)
{
  // Updates the filter with a measured distance in the zb direction using the
  float h[KC_STATE_DIM] = {0};
  arm_matrix_instance_f32 H = {1, KC_STATE_DIM, h};

  // Only update the filter if the measurement is reliable (\hat{h} -> infty when R[2][2] -> 0)
  if (fabs(this->R[2][2]) > 0.1 && this->R[2][2] > 0){
    float angle = fabsf(acosf(this->R[2][2])) - DEG_TO_RAD * (15.0f / 2.0f);
    if (angle < 0.0f) {
      angle = 0.0f;
    }
    //float predictedDistance = S[KC_STATE_Z] / cosf(angle);
    float predictedDistance = this->S[KC_STATE_Z] / this->R[2][2];
    float measuredDistance = tof->distance; // [m]

    // Custom code to avoid height jumps when passing over obstacle/gate
    if (waitCounterTakeoff < thresholdTakeoff){
      if(measuredDistance > desiredFlightHeight-0.1f && flag==0){
        waitCounterTakeoff += 1;
      }
    }
    else {
      if(flag==0 ){
        DEBUG_PRINT("Kalman Filter takeoff countdown completed\n");
        flag=1;
      }
      if (fabsf(desiredFlightHeight - measuredDistance) > heightDiffThreshold && waitCounterObstacle < thresholdObstacle) {
        waitCounterObstacle += 1;
        return;
      }
      else {
        waitCounterObstacle = 0;

      }
    }

    
    //Measurement equation
    //
    // h = z/((R*z_b)\dot z_b) = z/cos(alpha)
    h[KC_STATE_Z] = 1 / this->R[2][2];
    //h[KC_STATE_Z] = 1 / cosf(angle);

    // Scalar update
    kalmanCoreScalarUpdate(this, &H, measuredDistance-predictedDistance, tof->stdDev);
  }
}
