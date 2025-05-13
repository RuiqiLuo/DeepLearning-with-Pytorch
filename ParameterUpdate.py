import torch
N,D_in,H,D_out=64,1000,100,10
#随机输入和输出张量
x=torch.randn(N,D_in)
y=torch.randn(N,D_out)#这里为什么不是H
#使用nn包定义模型模型和损失函数
#这里Sequential的作用：
model=torch.nn.Sequential(
torch.nn.Linear(D_in,H),
torch.nn.ReLU(),
torch.nn.Linear(H,D_out),#怎么这里又变成了H，这里和输入和输出张量又有什么联系
)
#损失函数
loss_fn=torch.nn.MSELoss(reduction='sum')#.MSELoss(计算均方误差),reduction='sum'表示对每个样本的损失求和
#学习率
learning_rate=1e-4


#使用optim包定义优化器
#optim主要作用：根据反向传播计算得到的梯度，自动更新模型的参数
#优化方法选择：Adam
#Adam：结合了动量法，自适应学习率思想的优化算法
#其中又有两个重要的概念：Momentum和RMSprop
#其中Momentum模拟惯性：梯度会有加速度，防止来回震荡
#RMSprop为每个参数分配不同的学习率，避免学习率设置不合适导致问题
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)#.parameters()返回模型的所有参数，lr表示学习率,这里Adam是optim包里的方法?
for t in range(500):
    #前向传播
    y_pred=model(x)#这里的model(x)是怎么计算的?
    #计算损失
    loss=loss_fn(y_pred,y)#这里的loss_fn(y_pred,y)是怎么计算的?
    if t%100==0:
        print(t,loss.item())#.item()表示把张量转换成Python的数字，如果不转换会这么样
        print(t,loss)#这里设置一个来对比一下
    #使用反向传播之前，使用optimizer将更新的张量梯度清零（这里的本质是什么？）
    optimizer.zero_grad()#这里的zero_grad()是optim里的方法?
    #反向传播
    loss.backward()#这里计算的是什么
    #调用optimizer.step()方法来更新参数
    optimizer.step()
