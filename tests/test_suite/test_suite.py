import os
import subprocess
import profile_utils as prof

timeout = 300
cwd = os.getcwd()
filename = cwd+"/test_suite_results.txt"
f = open(filename, 'w')



print(cwd)

os.chdir(cwd+"/..")

cwd = os.getcwd()

print(cwd)

print("\n=====> ENTERING TEST SEQUENCE FOR MHSA.. <=====\n")
os.chdir("test_mhsa")

os.system("rm -r BUILD/")
cmd = "make clean get_golden all run STEP='FORWARD' > log.txt"
subprocess.check_output(cmd, shell=True, timeout=timeout)
prof.extract_performance("\nMHSA FORWARD check...\n", 0, filename)

os.system("rm -r BUILD/")
cmd = "make clean get_golden all run STEP='BACKWARD' > log.txt"
subprocess.check_output(cmd, shell=True, timeout=timeout)
prof.extract_performance("\nMHSA BACKWARD check...\n", 0, filename)


print("\n=====> ENTERING TEST SEQUENCE FOR IM2COL.. <=====\n")
os.chdir(os.getcwd()+"/../test_im2col")

os.system("rm -r BUILD/")
cmd = "make clean all run > log.txt"
subprocess.check_output(cmd, shell=True, timeout=timeout)
prof.extract_performance("\nim2col check...\n", 0, filename)
