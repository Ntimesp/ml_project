import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#import BSIF
from MRF import mrf
from CS_KDA import CS_KDA

def myimread(name):
    pic=mpimg.imread(name)
    return np.dot(pic[...,:3], [0.299, 0.587, 0.114])

def mylistdir(dir):
    files=[]
    for file in os.listdir(dir):
        if file.endswith('.jpg'):
            files.append(file)
    
    return files

paths=[os.path.join('LFW',"mismatch pairs"),os.path.join('LFW',"match pairs")]
for path in paths:
    dirs=os.listdir(path)
    for dir in dirs:
        di=os.path.join(path,dir)
        files=mylistdir(di)
        for file in files:
            
            target=myimread(os.path.join(di,file))
            tt=mrf(target)
            plt.imsave(os.path.join('test',dir+file),tt,cmap='gray')