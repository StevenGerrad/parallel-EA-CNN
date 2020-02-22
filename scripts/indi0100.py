"""
2020-02-22  18:18:29
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import data_loader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool
from thop import profile

from collections import defaultdict, OrderedDict
import json

from MadeData import MadeData


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()

        # conv and bn_relu layers
        self.globalPool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.layer_vertex = torch.nn.ModuleList()
        self.layer_vertex.append(None)

        self.layer_vertex.append(None)

        self.layer_vertex.append(torch.nn.Sequential(torch.nn.BatchNorm2d(5),torch.nn.ReLU(inplace=True)))

        self.layer_vertex.append(torch.nn.Sequential(torch.nn.Linear(10, 10)))

        self.layer_edge = torch.nn.ModuleList()
        self.layer_edge.append(None)

        self.layer_edge.append(None)

        self.layer_edge.append(torch.nn.Conv2d(3,5,kernel_size=(3,3),stride=pow(2, 0),padding=(1, 1)))

        self.layer_edge.append(torch.nn.Conv2d(3,5,kernel_size=(3,3),stride=pow(2, 0),padding=(1, 1)))



    def forward(self, input):
        block_h = input.shape[0]
        x0 = input
        x1 = x0
        x2 = self.layer_edge[3](x1)
        x2 = self.layer_vertex[2](x2)

        e2 = self.layer_edge[2](x0)

        e1 = x2

        x3 = torch.cat([e2,e1], dim=1)
        x3 = self.globalPool(x3)

        x3 = torch.squeeze(x3, 3)

        x3 = torch.squeeze(x3, 2)

        x3 = self.layer_vertex[3](x3)

        return x3


class TrainModel(object):
    def __init__(self, learning_rate):
        '''
        需传入: 学习率、data
        '''
        # trainloader, validate_loader = data_loader.get_train_valid_loader('/home/yanan/train_data', batch_size=128, augment=True, valid_size=0.1, shuffle=True, random_seed=2312390, show_sample=False, num_workers=1, pin_memory=True)
        #testloader = data_loader.get_test_loader('/home/yanan/train_data', batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        net = EvoCNNModel()
        cudnn.benchmark = True
        # net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        self.data = MadeData()
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        # self.trainloader = trainloader
        # self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)
        self.num_class = StatusUpdateTool.get_num_class()
        self.learning_rate = learning_rate

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, train_loader):
        self.net.train()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        loss_func = torch.nn.CrossEntropyLoss()

        running_loss = 0.0
        total = 0
        correct = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            output = self.net(b_x)  # cnn output
            idy = b_y.view(-1, 1)

            loss = loss_func(output, b_y)  # cross entropy loss
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # running_loss += loss.data[0]*labels.size(0)
            running_loss += loss.data.numpy()*b_y.size(0)
            _, predicted = torch.max(output.data, 1)
            total += b_y.size(0)
            # TODO: 经常是0.00
            correct += (predicted == b_y.data).sum()

        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))

    def test(self, epoch, testloader):
        accuracy, test_loss = self.Accuracy(self.net, testloader)
        # input = torch.randn(self.BATCH_SIZE, dna.input_size_channel, dna.input_size_height, dna.input_size_width)

        # TODO: 暂时不考虑关于FLOPS的计算
        # t = testloader[0][0].size()
        # input = torch.randn(t[0], t[1], t[2], t[3])
        # flops, params = profile(self.net, inputs=(input, ))
        # print('----- Accuracy: {:.6f} Flops: {:.6f}-----'.format(accuracy, flops))


        if accuracy > self.best_acc:
            self.best_acc = accuracy
            #print('*'*100, self.best_acc)
        self.log_record('Validate-Loss:%.3f, Acc:%.3f' % (test_loss, accuracy))

    def Accuracy(self, net, testloader):
        ''' https://blog.csdn.net/Arctic_Beacon/article/details/85068188 '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        test_loss = 0.0
        total = 0
        class_correct = list(0. for i in range(self.num_class))
        class_total = list(0. for i in range(self.num_class))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                BATCH_SIZE = images.shape[0]

                outputs = net(images)

                total += labels.size(0)
                loss = self.criterion(outputs, labels)
                # test_loss += loss.data[0]*labels.size(0)
                test_loss += loss.item()*labels.size(0)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(BATCH_SIZE):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # for i in range(self.N_CLASSES):
        #     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        return sum(class_correct) / sum(class_total), test_loss / total

    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        train_loader, testloader = self.data.CIFR10()
        for p in range(total_epoch):
            self.train(p, train_loader)
            self.test(total_epoch, testloader)
        return self.best_acc


class RunModel(object):
    def do_work(self, cpu_id, file_id, learning_rate):
        os.environ['CUDA_VISIBLE_DEVICES'] = cpu_id
        self.best_acc = 0.0
        try:
            m = TrainModel(learning_rate)
            m.log_record('Used CPU#%s, worker name:%s[%d]'%(cpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            self.best_acc = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.3f' % self.best_acc)
            # f = open('./populations/after_%s.json'%(file_id[4:6]), 'a+')
            # f.write('%s=%.5f\n'%(file_id, self.best_acc))
            # f.flush()
            # f.close()
            self.record_accuracy(file_id)

    def record_accuracy(self, file_id):
        file_name = file_id
        # self.log.info('%s has inherited the fitness as %.5f, no need to evaluate' %(file_name, indi.acc))
        # f = open('./populations/after_%s.json' % (file_name[4:6]), 'a+')
        # f.write('%s=%.5f\n' % (file_name, indi.acc))

        # 若没有after记录文件则初始化一个
        if os.path.exists('./populations/after_%s.json' % (file_name[4:6])) != True:
            after_dict = {
                'version': "1.0",
                'cache': [],
                'explain': {
                    'used': True,
                    'details': "this is for after evaluate",
                }
            }
            json_str = json.dumps(after_dict)
            with open('./populations/after_%s.json' % (file_name[4:6]), 'w') as json_file:
                json_file.write(json_str)

        # 记录训练结果
        f = open('./populations/after_%s.json' % (file_name[4:6]), 'r')
        info = json.load(f)

        individual = defaultdict(list)
        individual["file_name"] = file_name
        individual["accuracy"] = self.best_acc

        # TODO: 若上一次系统运行的结果还在则需要进行覆盖
        duplicate_indi = False
        for i in info["cache"]:
            if i["file_name"] == file_name:
                i["accuracy"] = self.best_acc
                duplicate_indi = True
                break
        if duplicate_indi == False:
            info["cache"].append(individual)

        json_str = json.dumps(info)
        with open('./populations/after_%s.json' % (file_name[4:6]), 'w') as json_file:
            json_file.write(json_str)