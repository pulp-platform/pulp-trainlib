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
import os
import shutil

# Copy related test folder into temp
def copy_test_folder_ci (test_id, ci_test_folder, test_folder):

    test_dest_folder = str(ci_test_folder)+"/temp/tests/ci_test_"+str(test_id)
    if not os.path.exists(test_dest_folder):
        os.mkdir(test_dest_folder)
    os.chdir(test_dest_folder)
    shutil.copytree(test_folder, test_dest_folder, dirs_exist_ok=True)


# Copy the trainlib into the suitable position
def copy_trainlib_ci (ci_test_folder, trainlib_folder):

    trainlib_dest_folder = str(ci_test_folder)+"/temp/lib"
    os.chdir(trainlib_dest_folder)
    shutil.copytree(trainlib_folder, trainlib_dest_folder, dirs_exist_ok=True)


