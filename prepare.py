import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from BSIF import BSIF 
from BSIF import train_filters
from MRF import mrf
from CS_KDA import CS_KDA

import threading

def myread_img(name):
        #读入指定路径的图片，并返回灰度图(还需加上预处理)
        pic=mpimg.imread(name)
        pic=np.dot(pic[...,:3], [0.299, 0.587, 0.114])
        
        return pic

class myThread (threading.Thread):
    def __init__(self,path1,path2,dirs,count):
        threading.Thread.__init__(self)
        self.path1=path1
        self.path2=path2
        self.dirs=dirs
        self.count=count

    def run(self):
        print(self.path1,self.count)
        for dir in self.dirs:
            temp=os.path.join(self.path1,dir)
            os.mkdir(os.path.join(self.path2,dir))
            files=os.listdir(temp)
            for name in files:
                pic=myread_img(os.path.join(self.path1,dir,name))
                pic=mrf(pic)
                res=BSIF(pic)
                np.save(os.path.join(self.path2,dir,name[:-4]),res)

# 用十个线程运行预处理
paths=[os.path.join("LFW","mismatch pairs"),os.path.join("LFW","match pairs")]
path2="test"
os.mkdir(path2)
for y,path1 in enumerate(paths):
    threads = []
    os.mkdir(os.path.join(path2,str(y)))
    dirs=os.listdir(path1)
    for i in range(10):
        thread=myThread(path1,os.path.join(path2,str(y)),dirs[i*160:(i+1)*160],i)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        print(thread.path1,thread.count,"over")