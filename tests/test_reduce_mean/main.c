#include "pmsis.h"
#include "net.h"


int main(void) {
    printf("\nHello there.\nConfiguring cluster..\n");

    // Configure cluster
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    struct pi_cluster_task cl_task;

    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cluster_dev, &cl_conf);

    if (pi_cluster_open(&cluster_dev)) {
        return -1;
    }

    printf("\nLaunching broadcast matmul evaluation...\n\n");
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, reduce_mean_test, NULL));

    printf("\nMatmul evaluation successfully terminated :)\n");
    pi_cluster_close(&cluster_dev);

    pmsis_exit(0);
}
