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

#ifndef _STATS_H
#define _STATS_H

#ifdef BOARD

#define INIT_STATS()  
    unsigned long _cycles = 0; \
    int id = 0;

#define PRE_START_STATS()  \
      pi_perf_conf((1<<PI_PERF_CYCLES)); 


#define START_STATS()  \
    pi_perf_stop(); \
    pi_perf_reset(); \
    pi_perf_start();

#define STOP_STATS() \
   pi_perf_stop(); \
    _cycles   = pi_perf_read (PI_PERF_CYCLES); \
    id = pi_core_id(); \
    printf("\n"); \
    printf("[%d] cycles = %lu\n", id, _cycles); 

#else

#ifdef STATS

#define INIT_STATS()  
    unsigned long _cycles = 0; \
    unsigned long _instr = 0; \
    unsigned long _tcdmcont = 0; \
    unsigned long _ldstall = 0; \
    unsigned long _imiss = 0; \
    int id = 0;

#define PRE_START_STATS()  \
      pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR) | (1<<PI_PERF_TCDM_CONT) | (1<<PI_PERF_LD_STALL) | (1<<PI_PERF_IMISS) ); 


#define START_STATS()  \
    pi_perf_stop(); \
    pi_perf_reset(); \
    pi_perf_start();

#define STOP_STATS() \
   pi_perf_stop(); \
      _cycles   = pi_perf_read (PI_PERF_CYCLES); \
      _instr    = pi_perf_read (PI_PERF_INSTR); \
    	_tcdmcont = pi_perf_read (PI_PERF_TCDM_CONT); \
    	_ldstall  = pi_perf_read (PI_PERF_LD_STALL); \
      _imiss    = pi_perf_read (PI_PERF_IMISS); \
    id = pi_core_id(); \
    printf("\n"); \
    printf("[%d] elapsed clock cycles = %lu\n", id, _cycles); \
    printf("[%d] number of instructions = %lu\n", id, _instr); \
    printf("[%d] TCDM contentions = %lu\n", id, _tcdmcont); \
    printf("[%d] load stalls = %lu\n", id, _ldstall); \
    printf("[%d] icache miss (clk cycles count) = %lu\n", id, _imiss); 

#else // STATS

#define INIT_STATS()
#define PRE_START_STATS()
#define START_STATS()
#define STOP_STATS()

#endif  // STATS


#endif 

#endif
