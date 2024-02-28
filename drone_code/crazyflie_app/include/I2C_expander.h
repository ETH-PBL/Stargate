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

#ifndef I2C_EXPANDER // include guard
#define I2C_EXPANDER

// i2c ADDRESS
#define I2C_EXPANDER_DEFAULT_I2C_ADDRESS            ((uint8_t)0x20)
// Registers
#define OUTPUT_PORT_REG_ADDRESS                     ((uint8_t)0x01)
#define INPUT_PORT_REG_ADDRESS                      ((uint8_t)0x00)
#define CONFIGURATION_REG_ADDRESS                   ((uint8_t)0x03)
#define POLARITY_INVERSION_REG_ADDRESS              ((uint8_t)0x02)
// Pins
#define LPN_BACKWARD_PIN                            ((uint8_t)1<<0)
#define I2C_RST_BACKWARD_PIN                        ((uint8_t)1<<1)
#define LPN_FORWARD_PIN                             ((uint8_t)1<<2)
#define I2C_RST_FORWARD_PIN                         ((uint8_t)1<<3)
#define LED_FORWARD_PIN                             ((uint8_t)1<<4)
#define LED_BACKWARD_PIN                            ((uint8_t)1<<5)
#define INTERRUPT_SENSE_FORWARD_PIN                 ((uint8_t)1<<6)
#define INTERRUPT_SENSE_BACKWARD_PIN                ((uint8_t)1<<7)

#define LPN_BACKWARD_PIN_NUM                        ((uint8_t)0)
#define I2C_RST_BACKWARD_PIN_NUM                    ((uint8_t)1)
#define LPN_FORWARD_PIN_NUM                         ((uint8_t)2)
#define I2C_RST_FORWARD_PIN_NUM                     ((uint8_t)3)
#define LED_FORWARD_PIN_NUM                         ((uint8_t)4)
#define LED_BACKWARD_PIN_NUM                        ((uint8_t)5)
#define INTERRUPT_SENSE_FORWARD_PIN_NUM             ((uint8_t)6)
#define INTERRUPT_SENSE_BACKWARD_PIN_NUM            ((uint8_t)7)


bool I2C_expander_set_register(uint8_t reg_address,uint8_t reg_value);
bool I2C_expander_get_register(uint8_t reg_address,uint8_t* reg_value);
bool I2C_expander_set_output_pin(uint8_t pin_number,bool pin_state);
bool I2C_expander_get_input_pin(uint8_t pin_number,bool* pin_state);
bool I2C_expander_initialize();

#endif /* I2C_EXPANDER */