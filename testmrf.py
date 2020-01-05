import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import shutil
from MRF import mrf

paths=[os.path.join("LFW","mismatch pairs"),os.path.join("LFW","match pairs")]
path2="testmrf"
if os.path.exists(path2):
    shutil.rmtree(path2)

os.mkdir(path2)

def myimread(name):
    pic=mpimg.imread(name)
    return np.dot(pic[...,:3], [0.299, 0.587, 0.114])

for y,path1 in enumerate(paths):
    dirs=os.listdir(path1)
    for dir in dirs:
        files=os.listdir(os.path.join(path1,dir))
        for file in files:
            name=os.path.join(path1,dir,file)
            pic=myimread(name)
            pic=mrf(pic)
            plt.imsave(os.path.join(path2,file),pic,cmap='gray')