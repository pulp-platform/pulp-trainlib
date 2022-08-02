// Tensor sizes
#define Tker_H_l1 Tker_W_l1

#define Tout_W_l1 ((int)(Tin_W_l1-Tker_W_l1+LPAD+RPAD+WSTR)/WSTR)
#define Tout_H_l1 ((int)(Tin_H_l1-Tker_H_l1+UPAD+DPAD+HSTR)/HSTR)

#define weight_init 0.1

#define PAD_BW (Tker_W_l1-1)

#define i2c_b_size (Tker_H_l1*Tker_W_l1*Tin_C_l1*(Tin_H_l1-Tker_H_l1+UPAD+DPAD+HSTR)/HSTR*(Tin_W_l1-Tker_W_l1+LPAD+RPAD+WSTR)/WSTR)
#define i2c_b_size_bw (Tker_H_l1*Tker_W_l1*Tout_C_l1*Tin_H_l1*Tin_W_l1)

// Tensor checksum definition
#define ABS(x) ((x)>0?(x):(-(x)))
#define CHECK_TOLERANCE 1e-3
#define ERROR_TOLERANCE 0.01

// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

void net_step ();