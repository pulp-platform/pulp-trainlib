'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini, Leonardo Ravaglia
'''


"""
UTILS FOR MATMUL OPTIMIZATION EVALUATION
"""

# Extracts profiling information from log file
def extract_performance (matmul, filename) :
    
    # Set to 1 to see the read values during the execution
    DEBUG = 0

    # Open the og file to find the performances of all matmuls
    f = open("log.txt", "r")
    Lines = f.readlines()
    # Phrases to be found 
    perf_text = '] cycles = '
    instr_text = '] instr = '
    ext_ld_text = '] ext load = '
    TCDM_text = '] TCDM cont = '
    ld_st_text = '] ld stall = '
    imiss_text = '] imiss = '
    error_text = 'Error at index:'
    # Entries
    performances = []
    instructions = []
    ext_loads = []
    TCDM_contentions = []
    load_stalls = []
    icache_misses = []
    error_flag = 0
    # FInd performance
    for idx, line in enumerate(Lines):
        if (line.find(perf_text) != -1) :
            index = line.find(perf_text)
            performances.append(int(line[index+len(perf_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(instr_text) != -1) :
            index = line.find(instr_text)
            instructions.append(int(line[index+len(instr_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(ext_ld_text) != -1) :
            index = line.find(ext_ld_text)
            ext_loads.append(int(line[index+len(ext_ld_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(TCDM_text) != -1) :
            index = line.find(TCDM_text)
            TCDM_contentions.append(int(line[index+len(TCDM_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(ld_st_text) != -1) :
            index = line.find(ld_st_text)
            load_stalls.append(int(line[index+len(ld_st_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(imiss_text) != -1) :
            index = line.find(imiss_text)
            icache_misses.append(int(line[index+len(imiss_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(error_text) != -1) :
            error_flag = 1
    f.close()

    # Print results to performance file
    f = open(filename, "a")
    if error_flag == 0 :
        f.write("\nMM {}  => cycles:\n{}".format(matmul, performances[0]))
        f.write("\ninstr = {}, ext_ld = {}, TCDM_cont = {}, ld_stalls = {}, imiss = {}\n".format(instructions[0], ext_loads[0], TCDM_contentions[0], load_stalls[0], icache_misses[0]))
    else:
        f.write("\nMM {}  \nCONTAINS ERRORS!!!".format(matmul))
    f.close()
    
    return



# Gets the names of the matmul algorithms from library file
def get_matmul_names (filename, matmul_group):

    name_list = []
    start_listing = False

    f = open(filename, "r")
    Lines = f.readlines()
    matmul_text = "matmul_type =="

    if matmul_group == "STANDARD":

        for idx, line in enumerate(Lines):
            if (line.find("STANDARD MATMULS") != -1):
                start_listing = True
            
            if (line.find("END STANDARD") != -1):
                start_listing = False
            
            if start_listing == True:
                if (line.find(matmul_text) != -1):
                    name_list.append(str(Lines[idx+1]))

    elif matmul_group == "DW":

        for idx, line in enumerate(Lines):
            if (line.find("DW MATMULS") != -1):
                start_listing = True
            
            if (line.find("END DW") != -1):
                start_listing = False
            
            if start_listing == True:
                if (line.find(matmul_text) != -1):
                    name_list.append(str(Lines[idx+1]))


    elif matmul_group == "DW_IN_GRAD":

        for idx, line in enumerate(Lines):
            if (line.find("DW_IN_GRAD MATMULS") != -1):
                start_listing = True
            
            if (line.find("END DW_IN_GRAD") != -1):
                start_listing = False
            
            if start_listing == True:
                if (line.find(matmul_text) != -1):
                    name_list.append(str(Lines[idx+1]))

    else:
        print("Invalid matmul name group selection!")
        exit()
    
    f.close()

    return name_list



# Sorts the profiled matmuls and appends them to file
def sort_best_performances (filename, matmul_group):

    f = open(filename, "r")
    Lines = f.readlines()
    search_text = 'MM '
    error_text = 'CONTAINS ERRORS!!!'
    # Entries
    algorithm = []
    performances = []
    for idx, line in enumerate(Lines):
        if (line.find(search_text) != -1):
            algorithm.append(str(line[0:5]))
            nextline = Lines[idx+1]
            if (nextline.find(error_text) != -1):
                performances.append(int('0'))
            else:
                performances.append(int(Lines[idx+1]))
    f.close

    # Create dictionary and sort
    algorithm_names = get_matmul_names("../mm_manager_list.txt", matmul_group)
    zip_iter = zip(algorithm_names, performances)
    def take_perf (e) :
        return e[-1]
    sorted_performances = sorted(zip_iter, key=take_perf, reverse=False)

    # Write sorted results
    f = open(filename, "a")
    f.write("\n------------------------------------------------\n\n")
    f.write("=====> BEST TO WORST <=====\n")
    f.write("(Results with 0 cycles contain errors)\n")
    name_idx = 0
    for alg, perf in sorted_performances:
        f.write("{} => {} cycles\n\n".format(alg, int(perf)))
    #f.write("------------------------------------------------\n")
    f.close()

    return


"""
UTILS FOR MULTIPLE NETWORK SIZES EVALUATION
"""

# Extracts profiling information from log file
def extract_size_performance (step, filename) :
    
    # Set to 1 to see the read values during the execution
    DEBUG = 0

    # Open the og file to find the performances of all matmuls
    f = open("log.txt", "r")
    Lines = f.readlines()
    # Phrases to be found 
    perf_text = '] cycles = '
    instr_text = '] instr = '
    ext_ld_text = '] ext load = '
    TCDM_text = '] TCDM cont = '
    ld_st_text = '] ld stall = '
    imiss_text = '] imiss = '
    error_text = 'Error at index:'
    # Entries
    performances = []
    instructions = []
    ext_loads = []
    TCDM_contentions = []
    load_stalls = []
    icache_misses = []
    error_flag = 0
    # FInd performance
    for idx, line in enumerate(Lines):
        if (line.find(perf_text) != -1) :
            index = line.find(perf_text)
            performances.append(int(line[index+len(perf_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(instr_text) != -1) :
            index = line.find(instr_text)
            instructions.append(int(line[index+len(instr_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(ext_ld_text) != -1) :
            index = line.find(ext_ld_text)
            ext_loads.append(int(line[index+len(ext_ld_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(TCDM_text) != -1) :
            index = line.find(TCDM_text)
            TCDM_contentions.append(int(line[index+len(TCDM_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(ld_st_text) != -1) :
            index = line.find(ld_st_text)
            load_stalls.append(int(line[index+len(ld_st_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(imiss_text) != -1) :
            index = line.find(imiss_text)
            icache_misses.append(int(line[index+len(imiss_text):-1]))
            if DEBUG == 1:
                print(line)

        if (line.find(error_text) != -1) :
            error_flag = 1
    f.close()

    if (len(performances) == 0):
        print("Performance not present, check if L1 memory is exceeded or convolution sizes are coherent with the input!!")
        exit()

    # Print results to performance file
    f = open(filename, "a")
    if error_flag == 0 :
        f.write("\n{} => cycles: {}".format(step, performances[0]))
        f.write(", instr = {}, ext_ld = {}, TCDM_cont = {}, ld_stalls = {}, imiss = {}\n".format(instructions[0], ext_loads[0], TCDM_contentions[0], load_stalls[0], icache_misses[0]))
    else:
        f.write("\n{} CONTAINS ERRORS!!!\n".format(step))
    f.close()
    
    return
