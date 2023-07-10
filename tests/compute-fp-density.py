'''
Copyright (C) 2023-2024 ETH Zurich and University of Bologna

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
Authors: Luca Valente
'''

from datetime import datetime
import sys
import re

prefixes = ("fadd.", "fsub.", "fmul.", 
            "fcvt.", "feq.", "flt.", "fle.", "fgt.", "fge.", "fsgn"
            )
prefixes2 = ( "fdiv.", "fsqrt.")

prefixes3 = ("fnmsub.", "fnmadd.", "fmadd.", "fmsub.", "vfadd.", "vfsub.", "vfmul.", 
            "vfmin.", "vfmax.", "vfclass.", "vfsgn",
            "vfeq.", "vflt.", "vfle.", "vfgt.", "vfge.",
            "vfcvt", "vfcpk", "vfavg.", "fmac.")

prefixes4 = ("vfdiv.", "vfsqrt.")

prefixes5 = ("lw", "sw", "p.lw", "p.sw", "c.lw", "c.sw", "lh", "sh", "p.lh", "p.sh", "c.lh", "c.sh")

prefixes6 = ( "vfmre.", "vfdot",  "vfmac.")

instr_counter = 0
fp_counter = 0
fp_counter2 = 0 
mem_counter = 0 
start_count = 0

filename = sys.argv[1]
runner = sys.argv[2]
gvsoc_trace = 0
if(runner=='gvsoc'):
   gvsoc_trace = 1

with open(filename, 'r') as f:

   # Read the file contents and generate a list with each line
   lines = f.readlines()
   for x in range(len(lines)):
      trace = lines[len(lines)-x-1].split()
      instruction = trace[4+3*(gvsoc_trace)]
      # Start from the bottom, when we execute pi_perf_stop()
      if (instruction=="csrrw"):
         if (trace[7+3*(gvsoc_trace)]=="0xcc1"):
            if "00000000" in trace[8+3*(gvsoc_trace)]:
               print("Start @%s" %(trace[1]))
               start_count = 1

      if start_count:
         # This is pi_perf_start(), we can exit
         if (instruction=="csrrw"):
            if (trace[7+3*(gvsoc_trace)]=="0xcc1"):
               if "00000003" in trace[8+3*(gvsoc_trace)]:
                  print("Stop @%s" %(trace[1]))
                  break

         else:
            instr_counter = instr_counter + 1
            if instruction.startswith(prefixes):
                fp_counter = fp_counter + 1
            elif instruction.startswith(prefixes2):
                fp_counter2 = fp_counter2 + 1
            elif instruction.startswith(prefixes3):
                fp_counter = fp_counter + 2
            elif instruction.startswith(prefixes4):
                fp_counter2 = fp_counter2 + 2
            elif instruction.startswith(prefixes5):
                mem_counter = mem_counter + 1
            elif instruction.startswith(prefixes6):
                print(instruction)
                fp_counter = fp_counter + 4


print("Total instructions:  %d" %(instr_counter))
print("FP instructions:  %d" %(fp_counter))
print("fdiv/fsqrt instructions: %d " %(fp_counter2))
print("mem instructions:  %d" %(mem_counter))
