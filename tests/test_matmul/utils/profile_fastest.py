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
Sort available matmuls from the fastest to the slowest
"""

# Set to 1 to see the read values during the execution
DEBUG = 0

# Open the og file to find the performances of all matmuls
f = open("log.txt", "r")
Lines = f.readlines()

# Phrases to be found 
alg_text = '-----> Profiling '
perf_text = '] cycles = '
size_text = 'Matmul sizes are:'
error_text = 'Error at index:'

# Entries
sizes = ''
data_type = ''
transp = ''
num_cores = ''
algorithm = []
performances = []

num_alg = 0
num_perf = 0
error_flag = 0

for idx, line in enumerate(Lines):
    if (line.find(size_text) != -1) :
        sizes = Lines[idx+1]
        data_type = Lines[idx+2]
        transp = Lines[idx+3]
        num_cores = Lines[idx+4]
        if DEBUG == 1:
            print(sizes)
            print(data_type)
            print(transp)
            print(num_cores)

    if (line.find(alg_text) != -1) :
        num_alg += 1
        index = line.find(alg_text)
        algorithm.append(str(line[index+len(alg_text):-2]))
        if DEBUG == 1:
            print(num_alg)
            print(line)

    if (line.find(perf_text) != -1) :
        num_perf += 1
        index = line.find(perf_text)
        performances.append(int(line[index+len(perf_text):-1]))
        if DEBUG == 1:
            print(num_perf)
            print(line)

    if (line.find(error_text) != -1) :
        error_flag = 1


# Check entries' correctness
if (num_alg != num_perf) :
    print("\nWRONG ENTRIES IN THE ORIGINAL FILE!\n")
    exit()
if (num_alg == 0) :
    print("\nNO ENTRIES IN THE ORIGINAL FILE!!\n")
    exit()

# Create dictionary and sort
zip_iter = zip(algorithm, performances)
def take_perf (e) :
    return e[-1]
sorted_performances = sorted(zip_iter, key=take_perf, reverse=False)

f.close()



# Print results to new file
f = open("fastest_matmul.txt", "w")

f.write('=====> MATMULS WITH BEST TO WORST PERFORMANCES <=====\n')
f.write('PROPERTIES:\n')
f.write(sizes)
f.write(data_type)
f.write(transp)
f.write(num_cores)
if (error_flag == 1):
    f.write("\n*** SOME MATMULS CONTAIN ERRORS, SEE LOG FILE!! ***\n")

# Write to output file with correct formatting (which depends on algorithm length)
f.write('\nFASTEST TO SLOWEST:\n')
alg_idx = 0
for algorithm, performance in sorted_performances:
    if alg_idx < 10:
        if len(algorithm) < 5 :
            f.write('[' + str(alg_idx) + ']  : ' + str(algorithm) + '\t\t\t\t\t(' + str(performance) + ' cycles)\n')
        elif len(algorithm) < 10:
            f.write('[' + str(alg_idx) + ']  : ' + str(algorithm) + '\t\t\t\t(' + str(performance) + ' cycles)\n')
        else:
            f.write('[' + str(alg_idx) + ']  : ' + str(algorithm) + '\t\t(' + str(performance) + ' cycles)\n')
    else:
        if len(algorithm) < 5:
            f.write('[' + str(alg_idx) + '] : ' + str(algorithm) + '\t\t\t\t\t(' + str(performance) + ' cycles)\n')
        elif len(algorithm) < 10:
            f.write('[' + str(alg_idx) + '] : ' + str(algorithm) + '\t\t\t\t(' + str(performance) + ' cycles)\n')
        else:
            f.write('[' + str(alg_idx) + '] : ' + str(algorithm) + '\t\t(' + str(performance) + ' cycles)\n')
    alg_idx += 1

f.close()
