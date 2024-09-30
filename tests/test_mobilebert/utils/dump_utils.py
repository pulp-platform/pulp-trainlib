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


import torch

def tensor_to_string(tensor):
	tensor_string = ''
	ndim = len(tensor.size())
	print("NDIM", ndim)

	if ndim == 1:
		sz0 = tensor.size()[0]
		for i in range(sz0):
			tensor_string += str(tensor[i].item())
			tensor_string += 'f, ';# if i < sz0-1 else 'f'

	elif ndim == 2:
		sz0 = tensor.size()[0]
		sz1 = tensor.size()[1]
		print('Sizes: ',sz0,sz1)
		for i in range(sz0):
			for j in range(sz1):
				tensor_string += str(tensor[i][j].item())
				tensor_string += 'f, ';# if (i*j) < (sz0-1)*(sz1-1) else 'f'

	elif ndim == 3:
		sz0 = tensor.size()[0]
		sz1 = tensor.size()[1]
		sz2 = tensor.size()[2]
		print('Sizes: ', sz0, sz1, sz2)
		for i in range(sz0):
			for j in range(sz1):
				for k in range(sz2):
					tensor_string += str(tensor[i][j][k].item())
					tensor_string += 'f, '; # if (i*j*k) < (sz0-1)*(sz1-1)*(sz2-1) else 'f'

	elif ndim == 4:
		sz0 = tensor.size()[0]
		sz1 = tensor.size()[1]
		sz2 = tensor.size()[2]
		sz3 = tensor.size()[3]
		print('Sizes: ', sz0, sz1, sz2, sz3)
		for i in range(sz0):
			for j in range(sz1):
				for k in range(sz2):
					for t in range(sz3):
						tensor_string += str(tensor[i][j][k][t].item())
						tensor_string += 'f, '; # if (i*j*k*t) < (sz0-1)*(sz1-1)*(sz2-1)*(sz3-1) else 'f'

	else:

		pass # FIXME to be implemented


	return tensor_string