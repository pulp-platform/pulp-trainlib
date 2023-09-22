// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

// Padded sizes
#define Tout_H (Tin_H+UPAD+DPAD)
#define Tout_W (Tin_W+RPAD+LPAD)

void net_step ();