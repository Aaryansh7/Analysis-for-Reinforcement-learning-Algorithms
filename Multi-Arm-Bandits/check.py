import numpy as np
import time
#print(np.random.normal(1,0.1))
var=np.random.normal(0, 0.01,10)
act=np.zeros(8)
act+=5
#print(var)
'''
#print(act)
start_time = time.time()
b=time.time()-start_time
a=np.log(b)
print(a)
print(b)
'''

act[0]=1
act[1]=4
act[2]=2
act[3]=5
act[4]=6
act[5]=2
act[6]=1
act[7]=3

p=np.argmax(act)
print(p)
