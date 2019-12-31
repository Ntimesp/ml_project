import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

M=10**9

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
        sum=0
        for i in range(n):
            pic1=self.pics[i]
            sum+=pic1
            for j in range(i,n):
                pic2=self.pics[j]
                self.K[i,j]=self.K[j,i]=self.kernel(pic1,pic2)
                
        self.mean=sum/n
        self.Kv=np.zeros((n,1))
        
        #K(x_i,\eta)
        for i in range(n):
            pic=self.read_img(os.path.join(self.path,self.files[i]))
            self.Kv[i]=self.kernel(self.mean,pic)
        
    def read_img(self,name):
        #读入指定路径的图片，并返回灰度图(还需加上预处理)
        pic=mpimg.imread(name)
        return np.dot(pic[...,:3], [0.299, 0.587, 0.114])
    
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
        mean=self.mean
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
                                
        likelihood1=self.test(self.K,self.Kv-Kv2[:n])
        likelihood2=self.test(K,Kv-Kv2)
        
        if(likelihood1<likelihood2):
            print(1)
        else:
            print(0)
            
    def test(self,K,diff):
        n=K.shape[0]
        y=np.zeros((n,1))
        y[:(n-1)]=-1/(n*(n-1))**0.5
        y[n-1]=((n-1)/n)**0.5
        return abs((np.linalg.pinv(K).dot(y)).T.dot(diff))

