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

from MadeData import MadeData


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generated_init


    def forward(self, x):
        #generate_forward
        

class TrainModel(object):
    def __init__(self):
        '''
        需传入: 学习率、data
        '''
        # trainloader, validate_loader = data_loader.get_train_valid_loader('/home/yanan/train_data', batch_size=128, augment=True, valid_size=0.1, shuffle=True, random_seed=2312390, show_sample=False, num_workers=1, pin_memory=True)
        #testloader = data_loader.get_test_loader('/home/yanan/train_data', batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        net = EvoCNNModel()
        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        # self.trainloader = trainloader
        # self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)

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
        # learning rate 学习率设置
        # if epoch ==0: lr = 0.01
        # if epoch > 0: lr = 0.1;
        # if epoch > 148: lr = 0.01
        # if epoch > 248: lr = 0.001
        # optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        # net = self.model_stack[dna.dna_cnt]
        # print("[decode].[", dna.dna_cnt, "]", net)
        optimizer = torch.optim.Adam(net.parameters(), lr=net.learning_rate)
        loss_func = torch.nn.CrossEntropyLoss()

        # train_loader, testloader = self.data.getData()
        accuracy = 0
        # training and testing
        for epoch in range(self.EPOCH):
            step = 0
            max_tep = int(60000 / train_loader.batch_size)

            # train_acc = .0
            # len_y = 0
            running_loss = 0.0
            total = 0
            correct = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output = net(b_x)  # cnn output
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
                correct += (predicted == b_y.data).sum()

        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))

    def test(self, epoch, testloader):
        accuracy, test_loss = self.Accuracy(net, testloader)
        # input = torch.randn(self.BATCH_SIZE, dna.input_size_channel, dna.input_size_height, dna.input_size_width)
        t = testloader[0][0].size()
        input = torch.randn(t[0], t[1], t[2], t[3])
        flops, params = profile(net, inputs=(input, ))
        # print('----- Accuracy: {:.6f} Flops: {:.6f}-----'.format(accuracy, flops))
        # dna.fitness = eval_acc / len_y
        dna.fitness = accuracy
        self.fitness_dir[dna.dna_cnt] = accuracy
        # print('')
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            #print('*'*100, self.best_acc)
        self.log_record('Validate-Loss:%.3f, Acc:%.3f' % (test_loss, accuracy))
    
    def Accuracy(self, net, testloader):
        ''' https://blog.csdn.net/Arctic_Beacon/article/details/85068188 '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        test_loss = 0.0
        total = 0
        class_correct = list(0. for i in range(self.N_CLASSES))
        class_total = list(0. for i in range(self.N_CLASSES))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)

                loss = self.criterion(outputs, labels)
                test_loss += loss.data[0]*labels.size(0)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(self.BATCH_SIZE):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # for i in range(self.N_CLASSES):
        #     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        return sum(class_correct) / sum(class_total), test_loss / total

    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        train_loader, testloader = self.data.getData()
        for p in range(total_epoch):
            self.train(p, train_loader)
            self.test(total_epoch, testloader)
        return self.best_acc


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        try:
            m = TrainModel()
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.3f'%best_acc)

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()
"""