import torch
import numpy as np

def TensorToArray(T, hwc):
    res=[]
    dim = len(T.size())

    if (dim == 1 ):
        for i in range(len(T)):
            res.append(T[i])
        return res

    if(dim == 3):
        if(hwc):
            for h in range(T.size(1)):
                for w in range(T.size(2)):
                    for c in range(T.size(0)):
                        res.append(float(T[c][h][w]))

        else:
            for c in range(T.size(0)):
                for h in range(T.size(1)):
                    for w in range(T.size(2)):
                        res.append(float(T[c][h][w]))
    return res


def WriteArray(array, name, f, d):
    l = len(array)
    name = str(name)
    f.write(f"\n{d} {name}[{l}] = ")
    f.write(" {")
    for i in range(l):
        if d == "fp16":
            f.write(f" {np.float16(array[i])}")
        else:
            f.write(f" {array[i]}")
        if(i != l-1):
            f.write(",")
    f.write("};\n")
