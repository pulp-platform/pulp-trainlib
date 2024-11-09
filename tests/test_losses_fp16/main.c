/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "net.h"

#define FREQ_FC 370
#define FREQ_CL 370
#define FREQ_PE 370

/*
*  DUMMY MAIN
*  Configures cluster, then calls net_step()
*/
int main (void) {

    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);

    printf("\n\nFC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
                pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    unsigned int GPIOs = 89;

    #if PROFILE_POWER == 1
    pi_pad_function_set(GPIOs, 1);
    pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
    pi_gpio_pin_write(GPIOs, 0);
    #endif

  printf("\nHello there.\nConfiguring cluster..\n");
  // Configure cluster
  struct pi_device cluster_dev;
  struct pi_cluster_conf cl_conf;
  struct pi_cluster_task cl_task;

  pi_cluster_conf_init(&cl_conf);
  pi_open_from_conf(&cluster_dev, &cl_conf);
  if (pi_cluster_open(&cluster_dev))
  {
      return -1;
  }

  printf("\nLaunching matmul evaluation...\n\n");
  pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, net_step, NULL));

  printf("\nOptimizer evaluation successfully terminated :)\n");
  pi_cluster_close(&cluster_dev);

  pmsis_exit(0);
}