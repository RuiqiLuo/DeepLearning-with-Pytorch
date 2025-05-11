import torch
#创建两层神经网络，实现前向和反向传播
dtype = torch.float
device=torch.device("cuda:0")
#N是批量大小，D_in是输入维度
#H是隐藏层维度，D_out是输出维度
N,D_in,H,D_out = 64,1000,100,1
#创建随机输入和输出数据
x=torch.randn(N,D_in,device=device,dtype=dtype)
y=torch.randn(N,D_out,device=device,dtype=dtype)
#随机初始化权重
w1=torch.randn(D_in,H,device=device,dtype=dtype)
w2=torch.randn(H,D_out,device=device,dtype=dtype)
learning_rate = 1e-6#10的-6次方
for t in range(500):
    #前向传播
    h=x.mm(w1)#.mm表示矩阵乘法，和.dot有什么区别？
    #环境:.dot是Numpy数组的方法,.mm是PyTorch张量的方法
    #维度:.dot可以处理任意维度的数组,.mm只能处理二维矩阵
    #以后要用再仔细查，不然到时候忘了
    h_relu=h.clamp(min=0)#.clamp(min=0)表示把小于0的数置为0
    y_pred=h_relu.mm(w2)
    #计算损失
    loss=(y_pred-y).pow(2).sum().item()#.pow(2)表示平方,.sum()表示求和,.item()表示把张量转换成Python的数字
    if t%100==0:
        print(t,loss)
    #反向传播
    grad_y_pred=2.0*(y_pred-y)#损失函数对预测输出的偏导数
    grad_w2=h_relu.t().mm(grad_y_pred)#这里是怎么算的？计算损失函数对w2的梯度,grad_y_pred是损失函数对预测输出的偏导数，h_relu是隐藏层的输出
    grad_h_relu=grad_y_pred.mm(w2.t())#为什么这里的矩阵转置在里面?
    grad_h=grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1=x.t().mm(grad_h)
    #使用梯度下降更新权重
    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2
