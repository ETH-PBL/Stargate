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

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "i2cdev.h"

#include "I2C_expander.h"

bool I2C_expander_set_register(uint8_t reg_address,uint8_t reg_value)
{
  bool status;
  status = i2cdevWriteReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,reg_address,1, &reg_value); 
  return status;
}
bool I2C_expander_get_register(uint8_t reg_address,uint8_t* reg_value)
{
  bool status;
  status = i2cdevReadReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,reg_address,1, reg_value); 
  return status;
}
bool I2C_expander_set_output_pin(uint8_t pin_number,bool pin_state)
{
    bool status;
    uint8_t output_reg_value;

    //read output current value
    status = i2cdevReadReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,OUTPUT_PORT_REG_ADDRESS,1, &output_reg_value); 
    if (status == false)
        return status;

    //check if the pin is in the same status // no need to rewrite it
    if ((output_reg_value &  1<<pin_number )== pin_state)
        return true;

    //mask selected pin
    if (pin_state == true)
        output_reg_value |= (1<<pin_number);
    else
        output_reg_value &= ~(1<<pin_number);

    //update reg value
    status = i2cdevWriteReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,OUTPUT_PORT_REG_ADDRESS,1, &output_reg_value); 
    if (status == false)
        return status;

    return status;
}
bool I2C_expander_toggle_output_pin(uint8_t pin_number)
{
    bool status;
    uint8_t output_reg_value;

    //read output current value
    status = i2cdevReadReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,OUTPUT_PORT_REG_ADDRESS,1, &output_reg_value); 
    if (status == false)
        return status;

    //mask selected pin
    if ((output_reg_value &  1<<pin_number )== false)
        output_reg_value |= (1<<pin_number);
    else
        output_reg_value &= ~(1<<pin_number);

    //update reg value
    status = i2cdevWriteReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,OUTPUT_PORT_REG_ADDRESS,1, &output_reg_value); 
    if (status == false)
        return status;

    return status;
}
bool I2C_expander_get_input_pin(uint8_t pin_number,bool* pin_state)
{
    bool status;
    uint8_t input_reg_value;

    //read input current value
    status = i2cdevReadReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,INPUT_PORT_REG_ADDRESS,1, &input_reg_value); 
    if (status == false)
        return status;

    //mask selected pin
    *pin_state = ((input_reg_value&(1<<pin_number))) ? true : false;

    return status;
}
bool I2C_expander_initialize()
{
    uint8_t reg_value;
    bool status;

    //set all outputs zero
    reg_value = 0x30; // 1-->on, 0-->off
    status = I2C_expander_set_register(OUTPUT_PORT_REG_ADDRESS,0x00);
    if (status == false)
        return status;

    //configure inversion all 0
    reg_value = 0x00; // 1-->inveritng, 0-->non-inveritng
    status = i2cdevWriteReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,POLARITY_INVERSION_REG_ADDRESS,1, &reg_value); 
    if (status == false)
        return status;
        
    //configure pins out/in
    reg_value = INTERRUPT_SENSE_BACKWARD_PIN|INTERRUPT_SENSE_FORWARD_PIN; // 1-->input, 0-->output
    status = i2cdevWriteReg8(I2C1_DEV,I2C_EXPANDER_DEFAULT_I2C_ADDRESS,CONFIGURATION_REG_ADDRESS,1, &reg_value); 
    if (status == false)
        return status;

    
    return status;
}