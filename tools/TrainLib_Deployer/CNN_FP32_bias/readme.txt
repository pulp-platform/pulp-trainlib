To compile the application, run "make clean get_golden all run > log.txt".
If running on a board (not GVSoC), add "APP_CFLAGS += -DBOARD" to the user section of the Makefile (profiling of cycles only).
To modify the hyperparameters (learning rate, epochs, batch size still not implemented), 
edit the variables inside "utils/GM.py".
