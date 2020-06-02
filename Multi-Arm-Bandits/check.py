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



a=np.zeros(2)
a+=1
b=np.zeros(2)
b+=2
c=[a,b]
for i in range(len(c)):
	list=c[i]
	print(list)