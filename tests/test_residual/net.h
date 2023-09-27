
#define I2C_SIZE  KER_SIZE*KER_SIZE*CI*(HI - KER_SIZE + 2*PAD_SIZE + 1)*(WI - KER_SIZE + 2*PAD_SIZE +1)


void net_step();
void prepare_data();
void forward();
void backward();
void PrintBlob(void * b, int step);


#define PROFILE

