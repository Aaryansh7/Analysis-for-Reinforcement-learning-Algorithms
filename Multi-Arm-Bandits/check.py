import numpy as np
import time
#print(np.random.normal(1,0.1))
var=np.random.normal(0, 0.01,10)
act=np.zeros(10)
act+=5
#print(var)
#print(act)
start_time = time.time()
b=time.time()-start_time
a=np.log(b)
print(a)
print(b)