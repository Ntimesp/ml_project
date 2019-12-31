import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def myimread(name):
    pic=mpimg.imread(name)
    return np.dot(pic[...,:3], [0.299, 0.587, 0.114])

def mylistdir(dir):
    files=[]
    for file in os.listdir(dir):
        if file.endswith('.jpg'):
            files.append(file)
    
    return files

def mrf(target):
    template=myimread('template/tem.jpg')
    m1,n1=template.shape
    m2,n2=target.shape
    maxs=100000000000
    for i in range(m2-m1):
        for j in range(n2-n1):
            diff=target[i:i+m1,j:j+n1]-template
            s=np.sum(diff*diff)
            if s<maxs:
                maxs=s
                mini=i
                minj=j

    return target[mini:mini+m1,minj:minj+n1]