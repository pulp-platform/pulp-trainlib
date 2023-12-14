To compile the application, run "make clean get_golden all run".
To modify the hyperparameters (learning rate, epochs, batch size still not implemented), edit the variables inside "utils/GM.py".

N.B: this project needs to have an L2 of at least 8 MB in GVSoC, please edit GVSoC's memory map to fit this requirement.
To do so, modify the following files:

1) pulp-sdk/rtos/pulpos/kernel/chips/pulp/link.ld
  L2           : ORIGIN = 0x1c000004, LENGTH = 0x007ffffc

2) pulp-sdk/tools/gap-configs/configs/config/pulp.json
    "l2": {
      "base": "0x1C000000",
      "size": "0x00800000",