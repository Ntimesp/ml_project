import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
from tqdm import tqdm
from BSIF import BSIF 
from BSIF import train_filters
from MRF import mrf
from CS_KDA import CS_KDA

import multiprocessing as threading

def myread_img(name):
        #读入指定路径的图片，并返回灰度图(还需加上预处理)
        pic=mpimg.imread(name)
        pic=np.dot(pic[...,:3], [0.299, 0.587, 0.114])
        
        return pic

def myThread(path1,path2,dirs,count):
    print(path1,count)
    for dir in dirs:
        temp=os.path.join(path1,dir)
        os.mkdir(os.path.join(path2,dir))
        files=os.listdir(temp)
        for name in files:
            pic=myread_img(os.path.join(path1,dir,name))
            pic=mrf(pic)
            res=BSIF(pic)
            np.save(os.path.join(path2,dir,name[:-4]),res)
            #plt.imsave(os.path.join(path2,dir,name),pic)
        
    print(path1,count,"over")

# 用二十个线程并行预处理
if __name__ == '__main__':
    t=time.time()
    paths=[os.path.join("dataset_hard")]
    path2="hard"
    os.mkdir(path2)
    for y,path1 in enumerate(paths):
        threads = []
        dirs=os.listdir(path1)
        for i in range(20):
            thread=threading.Process(target=myThread,args=(path1,os.path.join(path2),dirs[i*10:(i+1)*10],i))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    print("total time:",str(time.time()-t))