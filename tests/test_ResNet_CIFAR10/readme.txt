To compile the application, run "make clean get_golden all run".
To modify the hyperparameters (learning rate, epochs, batch size still not implemented), edit the variables inside "utils/GM.py".

N.B: this project needs to have an L2 of at least 8 MB in GVSoC, please edit GVSoC's memory map to fit this requirement.
To do so, copy & paste the content of "pulp-sdk-configs/"'s files in the respective files of your pulp.sdk (THIS VERSION IS PREFERABLE: https://github.com/pulp-platform/pulp-sdk/releases/tag/2021.09.15):

1) pulp-sdk/rtos/pulpos/kernel/chips/pulp/link.ld

2) pulp-sdk/tools/gap-configs/configs/config/pulp.json
