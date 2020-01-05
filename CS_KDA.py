import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tqdm import tqdm
from multiprocessing import Pool
import random
import shutil

M=10**7

class CS_KDA():
    def __init__(self):
        #初始化
        path='imposters'
        files=os.listdir(path)
        n=len(files)
        self.K=np.zeros((n,n))
        self.pics=[]

        for i in range(n):
            pic=self.read_img(os.path.join(path,files[i]))
            self.pics.append(pic)

        #计算K
        if __name__ == '__main__':
            for i in tqdm(range(n)):
                pic1=self.pics[i]
                for j in range(i,n):
                    pic2=self.pics[j]
                    self.K[i,j]=self.K[j,i]=self.kernel(pic1,pic2)
            np.save("K.npy",self.K)
        else:
            self.K=np.load("K.npy")

        #保存逆矩阵
        self.pinvK=np.linalg.pinv(self.K)
        #计算mean
        sum=0
        for i in range(n):
            pic1=self.pics[i]
            sum+=pic1

        self.mean=sum/n
        self.Kv=np.zeros((n,1))
        
        #K(x_i,\eta)
        for i in range(n):
            pic=self.read_img(os.path.join(path,files[i]))
            self.Kv[i]=self.kernel(self.mean,pic)


    def read_img(self,name):
        pic=np.load(name)
        #pic=plt.imread(name)

        return pic
    
    def kernel(self,pic1,pic2):
        #计算kernel变换后的内积
        temp=-np.sum((pic1-pic2)**2)/M
        return np.exp(temp)
    
    def predict(self,pic1,pic2):
        
        n=self.K.shape[0]
        
        #计算第二个类的K
        K=np.zeros((n+1,n+1))
        K[:n,:n]=self.K
        for i in range(n):
            temp=self.pics[i]
            K[i,n]=K[n,i]=self.kernel(pic1,temp)
            
        K[n,n]=self.kernel(pic1,pic1)
        
        #计算第二个类的KV
        mean=(n*self.mean+pic1)/(n+1)
        Kv=np.zeros((n+1,1))
        for i in range(n):
            temp=self.pics[i]
            Kv[i]=self.kernel(mean,temp)
                            
        Kv[n]=self.kernel(mean,pic1)
                                
        #计算pic2和两个类的相似度

        Kv2=np.zeros((n+1,1))
        for i in range(n):
            temp=self.pics[i]
            Kv2[i]=self.kernel(pic2,temp)
                                
        Kv2[n]=self.kernel(pic2,pic1)
                                
        likelihood1=self.test(self.pinvK,self.Kv-Kv2[:n])
        likelihood2=self.test(np.linalg.pinv(K),Kv-Kv2)
        
        if(likelihood1<likelihood2):
            return 1
        else:
            return 0
            
    def test(self,K,diff):
        n=K.shape[0]
        y=np.zeros((n,1))
        y[:(n-1)]=-1/(n*(n-1))**0.5
        y[n-1]=((n-1)/n)**0.5
        return abs((K.dot(y)).T.dot(diff))

if __name__ == '__main__':
    num1=1000
    path2='imposters'
    print("抽取"+str(2*num1)+"个imposters")
    try:
        os.mkdir(path2)
    except:
        shutil.rmtree(path2)
        os.mkdir(path2)

    for d in ["0"]:
        path1=os.path.join('test',d)
        
        dirs=os.listdir(path1)
        for dir in random.sample(dirs,num1):
            files=os.listdir(os.path.join(path1,dir))
            shutil.copyfile(os.path.join(path1,dir,files[0]),os.path.join(path2,files[0]))
            shutil.copyfile(os.path.join(path1,dir,files[1]),os.path.join(path2,files[1]))

    print("开始训练")
    CS_KDA()