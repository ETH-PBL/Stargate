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


/* Autotiler includes. */
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "main.h"
#include "gate_classifier_model_v51Kernels.h"
#include "gate_navigator_model_v22Kernels.h"
#include "gaplib/fs_switch.h"
#include "bsp/camera/himax.h"
#include "gaplib/ImgIO.h"
#include "pmsis/task.h"

#include "bsp/bsp.h"
#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif


#define TOF_ROWS 8
#define TOF_COLS 8
#define TOF_ROWS_CNN 21
#define TOF_COLS_CNN 21

// Camera
#define IMG_ROWS 168
#define IMG_COLS 168
#define X_OFFSET 76  // (320 - 168) / 2
#define Y_OFFSET 80  // (320 - 240) 

// Standardizer values
#define MEAN_IMAGE 0.2031f
#define MEAN_IMAGE_255_2pow9 26517
#define STD_IMAGE 0.0930f
#define STD_IMAGE_255 24
#define MEAN_TOF 2.7159f
#define STD_TOF 0.6062f

// Scales for quantizing CNN inputs
#define SCALE_QT_NAVIGATION_INPUT_IMAGE     0.06740149f
#define STD_IMAGE_255_SCALE_QT_NAVIGATION_INPUT_IMAGE     320//1.0f/1.598426335f*2^9
#define SCALE_QT_NAVIGATION_INPUT_TOF       0.03225412f
#define SCALE_QT_CLASSIFICATION_INPUT_IMAGE 0.06747100f
#define STD_IMAGE_255_SCALE_QT_CLASSIFICATION_INPUT_IMAGE 320//1.0f/1.59862815f*2^9
#define SCALE_QT_CLASSIFICATION_INPUT_TOF   0.03160720f

#define RESULT_BUFFERSIZE 3

AT_HYPERFLASH_FS_EXT_ADDR_TYPE gate_navigator_model_v22_L3_Flash = 0;
AT_HYPERFLASH_FS_EXT_ADDR_TYPE gate_classifier_model_v51_L3_Flash = 0;

//functions
void open_and_configure_cluster();
void open_and_configure_uart();
void open_and_configure_camera();
void cleanup_and_close();
void preprocess_img_tof_upsample();
void cluster_preprocessing();
void captured_image();
void main_loop();
void main_script(void);


// Raw inputs from camera
L2_MEM uint8_t img_buffer_1[IMG_ROWS*IMG_COLS];
L2_MEM uint8_t img_buffer_2[IMG_ROWS*IMG_COLS];
bool buffer_index;
L2_MEM uint8_t *last_buffer_img;

// Raw inputs from ToF over UART
L2_MEM uint16_t tof_buffer_mm_1[TOF_ROWS*TOF_COLS];
L2_MEM uint16_t tof_buffer_mm_2[TOF_ROWS*TOF_COLS];
L2_MEM uint16_t *cur_buffer_tof;

// ToF before resizing
L2_MEM int8_t tof_standardized_quantized_navigation[TOF_ROWS*TOF_COLS];
L2_MEM int8_t tof_standardized_quantized_classification[TOF_ROWS*TOF_COLS];

// CNN inputs, processed, standardized and quantized
L2_MEM int8_t img_standardized_quantized_navigation[IMG_ROWS*IMG_COLS];
L2_MEM int8_t tof_standardized_quantized_resized_navigation[TOF_ROWS_CNN*TOF_COLS_CNN];
L2_MEM int8_t img_standardized_quantized_classification[IMG_ROWS*IMG_COLS];
L2_MEM int8_t tof_standardized_quantized_resized_classification[TOF_ROWS_CNN*TOF_COLS_CNN];

// CNN outputs
L2_MEM int8_t output_navigation[1];
L2_MEM int8_t output_classification[1];

// Devices
static struct pi_device uart;
static struct pi_uart_conf uart_conf;
static struct pi_device camera;
static struct pi_himax_conf camera_conf;
struct pi_device cluster_dev;
struct pi_cluster_conf cl_conf;

// UART
int8_t data_to_send[RESULT_BUFFERSIZE];

//TASK STRUCTURES
L2_MEM pi_task_t task_uart,task_camera;//both tasks are used to inform the FC the arrival of data.
L2_MEM pi_task_t end_preprocessing,end_navigation,end_classification;//used to inform the FC the end of the executions in the cluster
struct pi_cluster_task task_preprocessing,task_navigation,task_classification;//tasks sent to the cluster that will be executed
L2_MEM pi_task_t end_peripherals;//used to inform the FC that bot camera image and ToF are received.


static void cluster_navigation()//activate the cluster for the navigation CNN
{
    __PREFIX1(CNN)(img_standardized_quantized_navigation, tof_standardized_quantized_resized_navigation, output_navigation);
}

static void cluster_classification()//activate the cluster for the classification CNN
{

    __PREFIX2(CNN)(img_standardized_quantized_classification, tof_standardized_quantized_resized_classification, output_classification);

}

void open_and_configure_cluster()//configure a cluster and defines the freq's
{
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	  printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIPH Frequency = %d Hz\n", pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));
}

void open_and_configure_uart()//configures and open the UART communication
{
    pi_uart_conf_init(&uart_conf);
    uart_conf.enable_tx = 1;
    uart_conf.enable_rx = 1;
    uart_conf.baudrate_bps = 921600;
    pi_open_from_conf(&uart, &uart_conf);
    if (pi_uart_open(&uart))
    {
        printf("Uart open failed !\n");
        pmsis_exit(-1);
    }
}

void open_and_configure_camera()//configure the camera
{
    pi_himax_conf_init(&camera_conf);
    camera_conf.roi.slice_en = 1;
    camera_conf.roi.x = X_OFFSET;
    camera_conf.roi.y = Y_OFFSET;
    camera_conf.roi.w = IMG_ROWS;
    camera_conf.roi.h = IMG_COLS;
    pi_open_from_conf(&camera, &camera_conf);
    if (pi_camera_open(&camera))
    {
        printf("Camera open failed !\n");
        pmsis_exit(-1);
    }
    /* Let the camera AEG work for 100ms */
    pi_camera_control(&camera, PI_CAMERA_CMD_AEG_INIT, 0);
}

void cleanup_and_close()//closes the camera
{
    pi_camera_close(&camera);
    pmsis_exit(0);
}

void preprocess_img_tof_upsample()//function performed in the cluster by each one of the cores
{
    uint8_t *cur_buffer_img= (buffer_index) ? img_buffer_1 : img_buffer_2;
    uint16_t *cur_buffer_tof= (buffer_index) ? tof_buffer_mm_1 : tof_buffer_mm_2;
    float pixel_intermediate;
    int32_t pixel_intermediate_int;
    int false_counter = 0;
    // Image: divide by 255, standardize, quantize img_buffer --> img_standardized_quantized_navigation
    for (int i = pi_core_id(); i < IMG_ROWS; i+=pi_cl_cluster_nb_cores ()) {
        for (int j = 0; j < IMG_COLS; j++) {
            // Access img_buffer in reverse order as the buffer needs to be flipped by 180 degree
            pixel_intermediate_int = (((int32_t)cur_buffer_img[IMG_ROWS*IMG_COLS-1-(j+IMG_COLS*i)])*0x200 - MEAN_IMAGE_255_2pow9);
            pixel_intermediate_int = (pixel_intermediate_int * STD_IMAGE_255_SCALE_QT_NAVIGATION_INPUT_IMAGE + 0x20000)>>18;
            img_standardized_quantized_navigation[j+IMG_COLS*i] = (int8_t)(pixel_intermediate_int);

            // we don't need to recompute, as with the fixed point values the quantization is the same for both NNs
            img_standardized_quantized_classification[j+IMG_COLS*i] = (int8_t)(pixel_intermediate_int);

        }
    }
    // ToF: divide by 1000 (convert to [m]), standardize, quantize tof_buffer_mm --> tof_standardized_quantized_navigation
    for (int i = pi_core_id(); i < TOF_COLS; i+=pi_cl_cluster_nb_cores ()) {
        for (int j = 0; j < TOF_ROWS; j++) {
            pixel_intermediate = ((cur_buffer_tof[j+TOF_COLS*i] / 1000.0f) - MEAN_TOF) / STD_TOF;
            tof_standardized_quantized_navigation[j+TOF_COLS*i] = (int8_t)round((pixel_intermediate / SCALE_QT_NAVIGATION_INPUT_TOF));
            tof_standardized_quantized_classification[j+TOF_COLS*i] = (int8_t)round((pixel_intermediate / SCALE_QT_CLASSIFICATION_INPUT_TOF));
        }
    }

    pi_cl_team_barrier();

    // Resize both tof_standardized_quantized_* --> tof_standardized_quantized_resized_*
    float resize_scale = (float)TOF_ROWS_CNN / (float)TOF_ROWS;
    int index_before_resize;
    // Upsample both ToF inputs with nearest neighbor interpolation
    for (int i = pi_core_id(); i < TOF_ROWS_CNN; i+=pi_cl_cluster_nb_cores ()) {
        for (int j = 0; j < TOF_COLS_CNN; j++) {
            index_before_resize = (int)floor((floor(j / resize_scale) + TOF_COLS * floor(i / resize_scale)));
            tof_standardized_quantized_resized_navigation[j+TOF_COLS_CNN*i] = tof_standardized_quantized_navigation[index_before_resize];
            tof_standardized_quantized_resized_classification[j+TOF_COLS_CNN*i] = tof_standardized_quantized_classification[index_before_resize];
        }
    }
}

void cluster_preprocessing(){//performed by core 0 of the cluster, performs the fork
    pi_cl_team_fork(0, preprocess_img_tof_upsample, NULL);//blocking
}

void captured_image(){//executed by the FC when a camera frame arrives
    pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
    //alert the main task that we have received both the UART and the camera frame
    pi_task_push(&end_peripherals);
}

void main_loop()
{
    //first ToF and image catched here to parallelize in the following loop cycle
    open_and_configure_uart();
    pi_uart_read(&uart, tof_buffer_mm_1, TOF_ROWS * TOF_COLS * sizeof(uint16_t));
    pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
    pi_camera_capture(&camera, img_buffer_1, IMG_ROWS * IMG_COLS);
    pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
    pi_time_wait_us(5000);
    pi_uart_close(&uart);
    


       int constructor_navigation_err = __PREFIX1(CNN_Construct)();
        if (constructor_navigation_err)
        {
            printf("Graph navigation constructor exited with error: %d\n(check the generated file gate_navigator_model_v22Kernels.c to see which memory have failed to be allocated)\n", constructor_navigation_err);
            pmsis_exit(-6);
        }

    gate_classifier_model_v51_L1_Memory = gate_navigator_model_v22_L1_Memory;

    int constructor_classification_err = __PREFIX2(CNN_Construct)();
        if (constructor_classification_err)
        {
            printf("Graph classification constructor exited with error: %d\n(check the generated file gate_navigator_model_v22Kernels.c to see which memory have failed to be allocated)\n", constructor_classification_err);
            pmsis_exit(-6);
        }

        

    while(true){
        //blocking notification triggered by the FC when both camera and UART are received
        pi_task_block(&end_peripherals);

        buffer_index = !buffer_index;//two buffers to parallelize computation and image/ToF catching 
        last_buffer_img = (buffer_index) ? img_buffer_2 : img_buffer_1;
        cur_buffer_tof = (buffer_index) ? tof_buffer_mm_2 : tof_buffer_mm_1;

        //We activate both the camera and UART communication, at the end of both the function "captured_image" is called end executed
        open_and_configure_uart();
        pi_uart_read_async(&uart,cur_buffer_tof, TOF_ROWS * TOF_COLS * sizeof(uint16_t),pi_task_block(&task_uart));

        pi_task_callback(&task_camera,captured_image,NULL);
        pi_camera_capture_async(&camera, last_buffer_img, IMG_COLS*IMG_ROWS,&task_camera);
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);

        // Image: divide by 255, standardize, quantize img_buffer --> img_standardized_quantized_navigation and img_standardized_quantized_classification
        // ToF: convert to [m], standardize, quantize tof_buffer_mm --> tof_standardized_quantized_navigation and tof_standardized_quantized_classification
        // Resize both tof_standardized_quantized_* --> tof_standardized_quantized_resized_*
        pi_cluster_task(&task_preprocessing,cluster_preprocessing,NULL );
        pi_cluster_send_task_async(&cluster_dev,&task_preprocessing,pi_task_block(&end_preprocessing));
        pi_task_wait_on(&end_preprocessing);

        pi_cluster_task(&task_navigation, (void (*)(void *))cluster_navigation, NULL);
        pi_cluster_task_stacks(&task_navigation, NULL, SLAVE_STACK_SIZE);
        pi_cluster_send_task_async(&cluster_dev, &task_navigation,pi_task_block(&end_navigation));

        pi_task_wait_on(&end_navigation);

        pi_cluster_task(&task_classification, (void (*)(void *))cluster_classification, NULL);
        pi_cluster_task_stacks(&task_classification, NULL, SLAVE_STACK_SIZE);
        pi_cluster_send_task_async(&cluster_dev, &task_classification,pi_task_block(&end_classification));
        
        pi_task_wait_on(&end_classification);
        
        //wait for the FC to retrieve both the camera image and the ToF
        //if more computation is needed, insert before this checkpoint.
        pi_task_wait_on(&task_uart);
        pi_task_wait_on(&end_peripherals);

        // UART communication
        data_to_send[0] = -128;  // Safety byte to check order on STM32 due to bug in UART
        data_to_send[1] = output_navigation[0];
        data_to_send[2] = output_classification[0];
        printf("%d\n", output_classification[0]);
        pi_uart_write(&uart, data_to_send, 3 * sizeof(int8_t));//try not to close it
        pi_uart_close(&uart);
        //This delay was introduced because two consecutive activation of the camera and/or the UART
        //create a bug in the system that will block the communication
        pi_time_wait_us(4700);
    }
}


void main_script(void)
{
    // Configure
    open_and_configure_cluster();

    open_and_configure_camera();

    buffer_index=0;//used to save the img and tof in different buffers


    main_loop();

    printf("Ended\n");
    cleanup_and_close();
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** IMAV CHALLENGE AIDECK MAIN ***\n\n");
    return pmsis_kickoff((void *) main_script);
}
