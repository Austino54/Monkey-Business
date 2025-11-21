import time
from gymnasium import spaces
import numpy as np
import asyncio
import threading

# print(time.localtime().tm_hour-6)
# print(time.time())
# print(time.ctime())
# print(time.localtime())
# print(time.gmtime())
# print(time.struct_time.tm_gmtoff)

# dataTypes = {
#     'np.int32': np.int32,
#     'np.int64': np.int64,
#     'np.uint32': np.uint32,
#     'np.uint16': np.uint16,
#     'np.float64': np.float64,
#     'np.float32': np.float32
# }

# print(dataTypes['np.int32'])

# class RunningTemp(threading.Thread):

arr = np.array([[2,3],[4,5]])
print(arr,'\n')
arr2 = np.array(arr)
print(arr2,'\n')
arr2[0,1] = 0
print(arr2,'\n')
print(arr,'\n')
arr = np.array(arr2)
print(arr)