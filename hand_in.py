import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import shutil
from tqdm import tqdm
from BSIF import BSIF 
from BSIF import train_filters
from MRF import mrf
from CS_KDA import CS_KDA

import random
import shutil
from multiprocessing import Pool

#从不同人物集合中随机抽取2*num1个构成imposters集合(num1最大1600))
#随机抽取num2组相同人物，num2组不同人物构成测试集（test1)

gap=10
def test(i):
    res=[]
    if i==0:
        for dir in tqdm(dirs[i*gap:(i+1)*gap]):
            files=os.listdir(os.path.join(path,dir))
            pic1=model.read_img(os.path.join(path,dir,files[0]))
            pic2=model.read_img(os.path.join(path,dir,files[1]))
            res.append(model.predict(pic1,pic2))
    else:
        for dir in dirs[i*gap:(i+1)*gap]:
            files=os.listdir(os.path.join(path,dir))
            pic1=model.read_img(os.path.join(path,dir,files[0]))
            pic2=model.read_img(os.path.join(path,dir,files[1]))
            res.append(model.predict(pic1,pic2))
    
    return res

#开始测试
print("初始化imposters类")
model=CS_KDA()
paths=[os.path.join("easy")]
ac=[]
for y,path in enumerate(paths):
    print("开始预测")
    dirs=os.listdir(path)
    dirs.sort()
    if __name__ == '__main__':
        with Pool(20) as p:
            re=p.map(test, range(20))

    res=[]
    for r in re:
        res.append(r)

    print(res)
    np.save(path+".npy",np.array(res))

import scipy.io as scio
import numpy as np

pre=np.load(path+".npy")
pre=pre.reshape((1,200))[0]
np.save(path+".npy",pre)
print(pre)



