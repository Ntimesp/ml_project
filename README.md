# 基于LFW数据集的人脸辨别程序

![introduction](introduction.png)

## 综述

我们复现了论文[MRF-Fusion-CSKDA](https://ieeexplore.ieee.org/document/6905848)，并在LFW数据集上进行了测试。

## 数据预处理

我们使用了MRF和MBSIF两种办法先后对数据进行了处理，其中MRF的目的是从图片中提取人脸的位置并对图片进行裁剪，MBSIF则是为了提取图片中和面部相关的主要信息。

### MRF(Markov Random Fields)

MRF是一个图模型，我们定义应组节点集合V和编辑和边集合C,以及能量函数
$$E(x)=\Theta_v(x)+\sum_{(x,y)\in C}\Theta_c(x,y)$$
需要极小化能量函数。

我们用一个模板图
![](tem.jpg)
在目标图片中找出最小化能量函数的边集合，即为面部，如下面所示
![](ans.jpg)
### MBSIF(Multiscale Binarised Statistical Image Features)

对于MRF处理后的图片，我们选取了一定规模的训练样本用来训练filters. 训练中使用的主要方法是ICA(Independent Component Analysis)，主要原理大致如下：

对于一个样本$X=(X_1,...,X_s)$,$(X_i \in \mathbb{R}^{d\times d})$, 首先先对其进行白化处理(white),即要求$V\in \mathbb{R}^{N\times d^2}$, 使得$VX$各个列向量之间协方差为0. 实际上，记$X$的协方差矩阵特征值分解为$C=EDE^T$，其中$C$为对角矩阵且对角元大小递减，$E$为正交方阵且每一列对应一个特征向量. 我们取了$V = [D^    {-1/2}E]_{1:N}$, 即取中括号内的矩阵的第一到第$N$列。

我们接下来处理过后的样本$S$, 利用独立成分分析的方法(FACTICA)求出解混矩阵$U$, 使得$Us$分量之间相互独立，其中$s$为$S$对应的随机向量。

到此，我们求出了$F = UV$，是一个$N\times d^2$阶的实矩阵，实际上我们取$N = 8$，再将每个行向量作为一个filter，用他们计算每个像素点对应的值，再用以下方法得到一个八位的二进制数:
$$\begin{aligned}
&r=UVp\\
&b_n = I_{x>0}(r_n)\\
&\text{bsif}(p)=\text{bin2dec}(b)\end{aligned}$$
依次选取$d=3,5,7,9,11,13,15,17$,我们一共得到了8组filters，每个filters作用在一个像素点（周围的矩形）上得到一个取值范围为$0-256$的数。

最后，我们需要将一张图片分成数个不交的矩形，通过以下方式计算出一个可以表征图片特征的向量：

$$
\begin{aligned}
&h_{j,s}=[h_{j,s}^0,...,h_{j,s}^{L-1}]\\
&h_{j,s}^i = \sum_{p_c\in G_j}\mathbf{1}_{\text{bsif}_s(p_c)=i}\\
&j\in [0,1,...,J\times J-1]\\
&s\in[1,2,...,Z]\\
& L=256
\end{aligned}
$$

## CS-KDA(Class-Specific Kernal Discriminant Analysis)

KDA的原理主要是先将几类样本用一个映射投影到像空间，再寻找一个方向$\alpha$使得在这个向量上这几类样本的区别最大。我们使用KDA的简化版本CS_KDA：给定2个人脸图片$x_1,x_2$, 先随机寻找一类包含人脸的图片$S$作为 imposter set,再用CS-KDA的方法求出$x_1$ 到$S$ 的距离和其到$S\cup\{x_2\}$ 集合的某种意义上(和KDA计算出的向量相关)距离，通过比较它们的大小来判断$x_1$和$x_2$是否是同一个人的人脸图像。具体的算法如下:

>CS_KDA Algorithm
>
>1. 设置初值$y=[\frac{-1}{\sqrt{m(m-1)}},...,\frac{-1}{\sqrt{m(m-1)}},\sqrt{\frac{m-1}{m}}]^T$
>2. 计算 imposter set $S =\{s_1,...,s_r\}$ 的第$j$个分块的kernal矩阵$K_j$和均值点$\gamma_j$, $j=1,...,J\times J-1$. 并求出$K=\sum_j K_j$.
>3. 计算 $S\cup \{x_2\}$的第$j$个分块的kernal矩阵$K'_j$和均值点$\gamma'_j$, $j=1,...,J\times J-1$. 并求出$K'=\sum_j K'_j$.
>4. 计算$K_\omega^-=\sum_j[\kappa(s_1^j,x_1^j)-\kappa(s_1^j,\gamma_j),...,\kappa(s_r^j,x_1^j)-\kappa(s_r^j,\gamma_j)]$ 和$\alpha_1=K^{-1}y$. 
>5. 计算$K_\omega^{'-}=\sum_j[\kappa(s_1^j,x_1^j)-\kappa(s_1^j,\gamma'_j),...,\kappa(s_r^j,x_1^j)-\kappa(s_1^j,\gamma'_j),\kappa(x_2^j,x_1^j)-\kappa(x_2^j,\gamma'_j)]$ 和$\alpha_2=K^{'-1}y$.
>6. 计算$p_1 = |\alpha_1^TK_\omega^-|$, $p_2 = |\alpha_2^TK_\omega^{'-}|$ 比较两者的大小。
## 实际过程
我们发现模型效果的好坏依赖于imposter set的大小，故我们实际上设置imposter set 的数量$r=2000$. 