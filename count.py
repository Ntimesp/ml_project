import time
import os
a=0
while(a<1):
    time.sleep(3)
    if(os.path.exists("test/0")):
        a=a+len(os.listdir("test/0"))

    if(os.path.exists("test/1")):
        a=a+len(os.listdir("test/1"))

    a=a/3200
    print(a)