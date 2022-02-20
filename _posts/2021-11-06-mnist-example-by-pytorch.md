---
layout: single
title:  "PyTorch MNIST 예제"
---

# Image Classification(MNIST Dataset)

```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input is 28x28
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        # feature map size is 14*14 by pooling
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        # feature map size is 7*7 by pooling 
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64) # reshape Variable
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train() # Module 클래스의 훈련 상태 여부 변경
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 역전파 단계를 실행하기 전에 변화도를 0으로 리셋
        output = model(data)
        loss = F.nll_loss(output, target) # Negative Log Likelihood
        loss.backward() # calc gradients
        optimizer.step() # update gradients
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval() # Module 클래스의 훈련 상태 여부 변경
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지(모델을 평가(evaluate)할 때 유용)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

import easydict

def main():
    # Training settings
    '''
    // ArgumentParser의 경우 Cmd prompt로 실행 시 인수로 줄 수 있음
    // 매크로를 이용해서 hyper parameter를 변경해 가면서 트레이닝하는데 적합
    // 여기서는 Google Colab으로 실행하는데 에러가 발생해서 easydict로 대체함
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    '''
    args = easydict.EasyDict({
        "batch_size": 100,
        "test_batch_size": 1000,
        "epochs" : 15,
        "lr": 0.005,
        "momentum" : 0.5,
        "no_cuda" : False,
        "seed" : 1,
        "log_interval" : 100,
        "save_model": False })

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # SGD = stochastic gradient descent (optionally with momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
    


    HBox(children=(FloatProgress(value=0.0, max=9912422.0), HTML(value='')))


    
    Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
    


    HBox(children=(FloatProgress(value=0.0, max=28881.0), HTML(value='')))


    
    Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
    


    HBox(children=(FloatProgress(value=0.0, max=1648877.0), HTML(value='')))


    
    Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


    HBox(children=(FloatProgress(value=0.0, max=4542.0), HTML(value='')))


    
    Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
    
    Processing...
    Done!
    

    /usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:143.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
    

    Train Epoch: 1 [0/60000 (0%)]	Loss: 2.299525
    Train Epoch: 1 [10000/60000 (17%)]	Loss: 0.867954
    Train Epoch: 1 [20000/60000 (33%)]	Loss: 0.363716
    Train Epoch: 1 [30000/60000 (50%)]	Loss: 0.320691
    Train Epoch: 1 [40000/60000 (67%)]	Loss: 0.185943
    Train Epoch: 1 [50000/60000 (83%)]	Loss: 0.184266
    
    Test set: Average loss: 0.2041, Accuracy: 9406/10000 (94%)
    
    Train Epoch: 2 [0/60000 (0%)]	Loss: 0.177712
    Train Epoch: 2 [10000/60000 (17%)]	Loss: 0.204368
    Train Epoch: 2 [20000/60000 (33%)]	Loss: 0.181901
    Train Epoch: 2 [30000/60000 (50%)]	Loss: 0.237086
    Train Epoch: 2 [40000/60000 (67%)]	Loss: 0.123869
    Train Epoch: 2 [50000/60000 (83%)]	Loss: 0.157136
    
    Test set: Average loss: 0.1225, Accuracy: 9649/10000 (96%)
    
    Train Epoch: 3 [0/60000 (0%)]	Loss: 0.138028
    Train Epoch: 3 [10000/60000 (17%)]	Loss: 0.084921
    Train Epoch: 3 [20000/60000 (33%)]	Loss: 0.124248
    Train Epoch: 3 [30000/60000 (50%)]	Loss: 0.111935
    Train Epoch: 3 [40000/60000 (67%)]	Loss: 0.120093
    Train Epoch: 3 [50000/60000 (83%)]	Loss: 0.128186
    
    Test set: Average loss: 0.0875, Accuracy: 9733/10000 (97%)
    
    Train Epoch: 4 [0/60000 (0%)]	Loss: 0.125996
    Train Epoch: 4 [10000/60000 (17%)]	Loss: 0.039393
    Train Epoch: 4 [20000/60000 (33%)]	Loss: 0.184345
    Train Epoch: 4 [30000/60000 (50%)]	Loss: 0.090806
    Train Epoch: 4 [40000/60000 (67%)]	Loss: 0.049221
    Train Epoch: 4 [50000/60000 (83%)]	Loss: 0.051194
    
    Test set: Average loss: 0.0669, Accuracy: 9797/10000 (98%)
    
    Train Epoch: 5 [0/60000 (0%)]	Loss: 0.124797
    Train Epoch: 5 [10000/60000 (17%)]	Loss: 0.113785
    Train Epoch: 5 [20000/60000 (33%)]	Loss: 0.047321
    Train Epoch: 5 [30000/60000 (50%)]	Loss: 0.079425
    Train Epoch: 5 [40000/60000 (67%)]	Loss: 0.060937
    Train Epoch: 5 [50000/60000 (83%)]	Loss: 0.056222
    
    Test set: Average loss: 0.0572, Accuracy: 9826/10000 (98%)
    
    Train Epoch: 6 [0/60000 (0%)]	Loss: 0.038805
    Train Epoch: 6 [10000/60000 (17%)]	Loss: 0.035069
    Train Epoch: 6 [20000/60000 (33%)]	Loss: 0.020468
    Train Epoch: 6 [30000/60000 (50%)]	Loss: 0.110463
    Train Epoch: 6 [40000/60000 (67%)]	Loss: 0.037852
    Train Epoch: 6 [50000/60000 (83%)]	Loss: 0.059460
    
    Test set: Average loss: 0.0515, Accuracy: 9842/10000 (98%)
    
    Train Epoch: 7 [0/60000 (0%)]	Loss: 0.037932
    Train Epoch: 7 [10000/60000 (17%)]	Loss: 0.015941
    Train Epoch: 7 [20000/60000 (33%)]	Loss: 0.048738
    Train Epoch: 7 [30000/60000 (50%)]	Loss: 0.061316
    Train Epoch: 7 [40000/60000 (67%)]	Loss: 0.067191
    Train Epoch: 7 [50000/60000 (83%)]	Loss: 0.113635
    
    Test set: Average loss: 0.0464, Accuracy: 9853/10000 (99%)
    
    Train Epoch: 8 [0/60000 (0%)]	Loss: 0.044765
    Train Epoch: 8 [10000/60000 (17%)]	Loss: 0.034175
    Train Epoch: 8 [20000/60000 (33%)]	Loss: 0.069408
    Train Epoch: 8 [30000/60000 (50%)]	Loss: 0.014703
    Train Epoch: 8 [40000/60000 (67%)]	Loss: 0.036426
    Train Epoch: 8 [50000/60000 (83%)]	Loss: 0.031561
    
    Test set: Average loss: 0.0490, Accuracy: 9840/10000 (98%)
    
    Train Epoch: 9 [0/60000 (0%)]	Loss: 0.008522
    Train Epoch: 9 [10000/60000 (17%)]	Loss: 0.024787
    Train Epoch: 9 [20000/60000 (33%)]	Loss: 0.050010
    Train Epoch: 9 [30000/60000 (50%)]	Loss: 0.026107
    Train Epoch: 9 [40000/60000 (67%)]	Loss: 0.032827
    Train Epoch: 9 [50000/60000 (83%)]	Loss: 0.048028
    
    Test set: Average loss: 0.0413, Accuracy: 9870/10000 (99%)
    
    Train Epoch: 10 [0/60000 (0%)]	Loss: 0.069179
    Train Epoch: 10 [10000/60000 (17%)]	Loss: 0.021759
    Train Epoch: 10 [20000/60000 (33%)]	Loss: 0.020901
    Train Epoch: 10 [30000/60000 (50%)]	Loss: 0.019622
    Train Epoch: 10 [40000/60000 (67%)]	Loss: 0.022783
    Train Epoch: 10 [50000/60000 (83%)]	Loss: 0.014935
    
    Test set: Average loss: 0.0378, Accuracy: 9870/10000 (99%)
    
    Train Epoch: 11 [0/60000 (0%)]	Loss: 0.058914
    Train Epoch: 11 [10000/60000 (17%)]	Loss: 0.031309
    Train Epoch: 11 [20000/60000 (33%)]	Loss: 0.011046
    Train Epoch: 11 [30000/60000 (50%)]	Loss: 0.052608
    Train Epoch: 11 [40000/60000 (67%)]	Loss: 0.003974
    Train Epoch: 11 [50000/60000 (83%)]	Loss: 0.020378
    
    Test set: Average loss: 0.0457, Accuracy: 9846/10000 (98%)
    
    Train Epoch: 12 [0/60000 (0%)]	Loss: 0.036000
    Train Epoch: 12 [10000/60000 (17%)]	Loss: 0.033474
    Train Epoch: 12 [20000/60000 (33%)]	Loss: 0.023501
    Train Epoch: 12 [30000/60000 (50%)]	Loss: 0.046775
    Train Epoch: 12 [40000/60000 (67%)]	Loss: 0.033960
    Train Epoch: 12 [50000/60000 (83%)]	Loss: 0.076243
    
    Test set: Average loss: 0.0354, Accuracy: 9878/10000 (99%)
    
    Train Epoch: 13 [0/60000 (0%)]	Loss: 0.025480
    Train Epoch: 13 [10000/60000 (17%)]	Loss: 0.005860
    Train Epoch: 13 [20000/60000 (33%)]	Loss: 0.012789
    Train Epoch: 13 [30000/60000 (50%)]	Loss: 0.020804
    Train Epoch: 13 [40000/60000 (67%)]	Loss: 0.008422
    Train Epoch: 13 [50000/60000 (83%)]	Loss: 0.056748
    
    Test set: Average loss: 0.0350, Accuracy: 9886/10000 (99%)
    
    Train Epoch: 14 [0/60000 (0%)]	Loss: 0.014957
    Train Epoch: 14 [10000/60000 (17%)]	Loss: 0.014809
    Train Epoch: 14 [20000/60000 (33%)]	Loss: 0.022703
    Train Epoch: 14 [30000/60000 (50%)]	Loss: 0.015825
    Train Epoch: 14 [40000/60000 (67%)]	Loss: 0.041507
    Train Epoch: 14 [50000/60000 (83%)]	Loss: 0.014923
    
    Test set: Average loss: 0.0321, Accuracy: 9902/10000 (99%)
    
    Train Epoch: 15 [0/60000 (0%)]	Loss: 0.027764
    Train Epoch: 15 [10000/60000 (17%)]	Loss: 0.037805
    Train Epoch: 15 [20000/60000 (33%)]	Loss: 0.011150
    Train Epoch: 15 [30000/60000 (50%)]	Loss: 0.018302
    Train Epoch: 15 [40000/60000 (67%)]	Loss: 0.025012
    Train Epoch: 15 [50000/60000 (83%)]	Loss: 0.009237
    
    Test set: Average loss: 0.0326, Accuracy: 9897/10000 (99%)
    
    
