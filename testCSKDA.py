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

num2=360



print("各抽取"+str(num2)+"组正负样本")
path1='test'
paths=[os.path.join(path1,'0'),os.path.join(path1,'1')]
path2='test1'

try:
    os.mkdir(path2)
except:
    shutil.rmtree(path2)
    os.mkdir(path2)

for y,path in enumerate(paths):
    dirs=os.listdir(path)
    n=0
    for dir in random.sample(dirs,num2):
        n=n+1
        shutil.copytree(os.path.join(path,dir),os.path.join(path2,str(y),str(n)))
        
from tqdm import tqdm

gap=5
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
paths=[os.path.join("test1","0"),os.path.join("test1","1")]
ac=[]
for y,path in enumerate(paths):
    print("开始预测"+str(y)+"集")
    dirs=os.listdir(path)
    if __name__ == '__main__':
        with Pool(12) as p:
            re=p.map(test, range(12))
    print(re)
    res=[]
    for r in re:
        res.append(r)

    #print(res)
    res=np.array(res)
    print("预测"+str(y)+"集正确率:",np.sum(res==y)/res.size)
    ac.append(np.sum(res==y)/res.size)
print("总正确率：",(ac[0]+ac[1])/2)