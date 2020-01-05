import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tqdm import tqdm
from multiprocessing import Pool
import random
import shutil

M=10**4
num_block = 42 #分块的数量#

class CS_KDA():
    def __init__(self):
        #初始化
        path='imposters'
        files=os.listdir(path)
        self.n=n=len(files)
        self.K=np.zeros((n,n))
        self.mean = []
        self.pics=[]

        for i in range(n):
            pic=np.load(os.path.join(path,files[i]), allow_pickle=True)
            self.pics.append(pic)

        #计算K
        if __name__ == '__main__':
            with Pool(12) as p:
                re=p.map(self.calblock, range(num_block))

            for Ki in re:
                self.K+=Ki
            
            np.save("K.npy",self.K)
        else:
            self.K=np.load("K.npy")

        for m in range(num_block):    
            sum=0
            for i in range(n):
                pic1=self.pics[i][:,m]
                sum+=pic1

            self.mean.append(sum/n)
        self.Kv=np.zeros((n,1))
        
        self.pinvK=np.linalg.pinv(self.K)

        #K(x_i,\eta)
        for i in range(n):
            for m in range(num_block):
                self.Kv[i]+=self.kernel(self.mean[m],self.pics[i][:,m])

    def calblock(self,m):
            print("start block",m)
            n=self.n
            K=np.zeros((n,n))
        
            for i in range(n):
                pic1=self.pics[i][:,m]
                for j in range(i,n):
                    pic2=self.pics[j][:,m]
                    K[i,j]=K[j,i]=self.kernel(pic1,pic2)

            print("finish block",m)
            return K

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
        for m in range(num_block):
            for i in range(n):
                temp=self.pics[i][:,m]
                K[i,n]+=self.kernel(pic1[:,m],temp)
                K[n,i]+=self.kernel(pic1[:,m],temp)
            
            K[n,n]+=self.kernel(pic1[:,m],pic1[:,m])
        
        #计算第二个类的KV
        mean = []
        for m in range(num_block):
            sum = self.mean[m]*n + pic1[:,m]
            mean.append(sum/(n+1))

        Kv=np.zeros((n+1,1))
        Kv[:n]=self.Kv
        for m in range(num_block):
            Kv[n]+=self.kernel(mean[m],pic1[:,m])
        
        #计算pic2和两个类的相似度

        Kv2=np.zeros((n+1,1))
        for m in range(num_block):
            for i in range(n):
                temp=self.pics[i]
                Kv2[i]+=self.kernel(pic2[:,m],temp[:,m])

            Kv2[n]+=self.kernel(pic2[:,m],pic1[:,m])


        likelihood1=self.test(self.pinvK,self.Kv-Kv2[:n])
        likelihood2=self.test(np.linalg.pinv(K),Kv-Kv2)
        if(likelihood2-likelihood1>0):
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
    num1=400
    path2='imposters'
    print("抽取"+str(4*num1)+"个imposters")
    try:
        os.mkdir(path2)
    except:
        shutil.rmtree(path2)
        os.mkdir(path2)

    for d in ["0","1"]:
        path1=os.path.join('test',d)
        
        dirs=os.listdir(path1)
        for dir in random.sample(dirs,num1):
            files=os.listdir(os.path.join(path1,dir))
            shutil.copyfile(os.path.join(path1,dir,files[0]),os.path.join(path2,files[0]))
            shutil.copyfile(os.path.join(path1,dir,files[1]),os.path.join(path2,files[1]))

    print("开始训练")
    CS_KDA()