# from __future__ import print_function


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

print("Pytorch version is: "+torch.__version__)
print("Torch CUDA is available == "+str(torch.cuda.is_available()))
torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 2 # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出
class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output


def dataloader():
    # 下载mnist手写数据集
    train_data = torchvision.datasets.MNIST(
        root='./data/',  # 保存或提取的位置  会放在当前文件夹中
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
    )

    test_data = torchvision.datasets.MNIST(
        root='./data/',
        train=False  # 表明是测试集
    )

    return train_data,test_data


def get_text_data(test_data):
    # 进行测试
    # 为节约时间，测试时只测试前2000个
    test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
    # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
    # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
    test_y = test_data.test_labels[:2000]
    return test_x,test_y

def get_cnn_net():
    # 加载网络到 gpu
    cnn = CNN()
    print(cnn)
    return cnn

def train(train_data,test_data):
    # 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
    # Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 是否打乱数据，一般都打乱
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取测试数据集
    test_x,test_y = get_text_data(test_data)
    test_x = test_x.to(device)

    # 获取 cnn 网络
    cnn = get_cnn_net()
    
    #将网络加载到gpu
    print(device)
    cnn = cnn.to(device)

    # 优化器选择Adam
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
    # 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差
    # 开始训练
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
            
            # 加载数据到gpu
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = cnn(b_x)  # 先将数据放到cnn中计算output
            ### 梯度下降算法 ###
            loss = loss_func(output, b_y)  # 损失函数，输出和真实标签的loss，二者位置不可颠倒
            
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度（权重更新）
            ### 梯度下降算法 ###
            # 每 50step 输出一次预测结果
            if step % 50 == 0:
                test_output = cnn(test_x)
                test_output = test_output.to('cpu')
                loss = loss.to('cpu')
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, 'setp: ',step,'| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    #保存模型
    torch.save(cnn.state_dict(), 'test.pkl') # 只保存了模型参数


# 测试
def predict(test_data):
    # 获取测试数据集
    test_x,test_y = get_text_data(test_data)
    # 获取cnn网络
    cnn = get_cnn_net()
    # 加载模型
    cnn.load_state_dict(torch.load('test.pkl'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #将模型加载到gpu上
    cnn.to(device)
    # 设置为推理模式
    cnn.eval()
    # 测试的个数
    testNum = 10
    # 输入的测试集
    inputs = test_x[:testNum]
    
    #将数据加载到gpu上
    inputs = inputs.to(device)

    # 输出的结果
    test_output = cnn(inputs)

    # 将数据加载回内存
    inputs = inputs.to('cpu')
    test_output = test_output.to('cpu')

    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')  # 打印识别后的数字
    print(test_y[:testNum].numpy(), 'real number')
    # 输出图片
    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)
    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    cv2.imshow('win', img)  # opencv显示需要识别的数据图片
    key_pressed = cv2.waitKey(0)



if __name__ == "__main__":
    # 数据加载
    train_data,test_data = dataloader()
    # 训练（先训练再预测）
    # train(train_data,test_data)
    # 预测（预测前将训练注释掉，否则会再训练一遍）
    predict(test_data)
 