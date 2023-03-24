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

/*
*  DUMMY MAIN
*  Configures cluster, then calls net_step()
*/
int main (void) {


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

  printf("\nLaunching training procedure...\n");
  pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, net_step, NULL));

  printf("Net training successful!\n");
  pi_cluster_close(&cluster_dev);

  pmsis_exit(0);
}
