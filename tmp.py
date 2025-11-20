import time
from gymnasium import spaces
import numpy as np
import asyncio

# print(time.localtime().tm_hour-6)
# print(time.time())
# print(time.ctime())
# print(time.localtime())
# print(time.gmtime())
# print(time.struct_time.tm_gmtoff)

# async def waiting():
#     await asyncio.sleep(3)
#     print("Finished coroutine :)")


# print("Starting coroutine...")
# asyncio.run(waiting())
# print("Passed the function... :/")


dataTypes = {
    'np.int32': np.int32,
    'np.int64': np.int64,
    'np.uint32': np.uint32,
    'np.uint16': np.uint16,
    'np.float64': np.float64,
    'np.float32': np.float32
}

print(dataTypes['np.int32'])