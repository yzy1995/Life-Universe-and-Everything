import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Trainer():
    def __init__(self, model, criterion='cross-entropy', optimizer='sgd', max_iter=3):
        assert model.parameters()!=None
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.max_iter = max_iter

    def fit(self, X):
        assert isinstance(X, torch.utils.data.dataloader.DataLoader)

        for epoch in range(1, self.max_iter+1):
            print('================\nEpoch: %d\n' % (epoch,))
            running_loss = 0.0
            for i, xi in enumerate(X, 1):
                # get input
                inputs, labels = xi
                inputs, labels = Variable(inputs), Variable(labels)

                # zero grads
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # backward
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]

                # log
                if i%2000==0:
                    print('Epoch %d\tBatch %d loss: %.3f' % (epoch, i, running_loss/2000.0))
                    running_loss = 0.0
        return self.model

    def predict(self, inputs):
        '''Return overall accuracy & accuracy for each class.'''
        assert isinstance(inputs, torch.utils.data.dataloader.DataLoader)

        correct, total = 0, 0
        class_correct, class_total = [0 for i in range(10)], [0 for j in range(10)]
        for xi in inputs:
            images, labels = xi
            images = Variable(images)
            outputs = self.model(images)
            _, y_preds = torch.max(outputs.data, 1)
            c = (y_preds == labels).squeeze()

            # accuracy for each class
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

            # overall accuracy
            correct += c.sum()
            total += labels.size(0)

        accuracy = 100 * correct / total
        class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
        return accuracy, class_accuracy

def load(batch_size=100):
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

