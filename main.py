import time
from util import Data
from model_trying import *
import os
import argparse
import pickle
import numpy
from numpy import random
import torch
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: RetailRocket/diginetica/Nowplaying/Tmall/yoochoose1_64/Lastfm')
parser.add_argument('--epoch', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('--beta', type=float, default=1.0, help='ssl task maginitude')#0.01
parser.add_argument('--lam', type=float, default=1.0, help='diff task maginitude')#0.005
parser.add_argument('--order', type=int, default=1, help='the order of multi grain')
parser.add_argument('--use_tar_bias', type=bool, default=False, help='is or not use tar multigrain')
parser.add_argument('--dropout', type=float, default=0, help='dropout')#0.005
parser.add_argument('--device', type=int, default=0, help='GPU-divice')
opt = parser.parse_args()
seed_torch(42)
torch.cuda.set_device(opt.device)
def main():
    train_data = pickle.load(open('/home/hzz/Gim/datasets/' + opt.dataset + '/train.txt', 'rb'))#351268 turple([userclick],[next_item]) userclick_len <belong to>[1,39]
    test_data = pickle.load(open('/home/hzz/Gim/datasets/' + opt.dataset + '/test.txt', 'rb'))#351268 turple([userclick],[next_item])
    all_train = pickle.load(open('/home/hzz/Gim/datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))#65285 list
    print(opt)
    # item = []
    # for session in train_data[0]:
    #     for i in session:
    #         item.append(i)
    # for session in all_train:
    #     for i in session:
    #         item.append(i)
    # print(min(item),max(item))
    # print(min(train_data[1]),max(train_data[1]))
    # print()
    # input()
    if opt.dataset == 'diginetica' or opt.dataset == 'diginetica2':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'RetailRocket':
        n_node = 36968
    elif opt.dataset == 'yoochoose1_64':
        n_node = 37483
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    elif opt.dataset == "Lastfm":
        n_node = 38615
    elif opt.dataset == "Gowalla":
        n_node = 29510
    else:
        n_node = 309
    train_data = Data(train_data,all_train, shuffle=False, n_node=n_node)
    test_data = Data(test_data,all_train, shuffle=False, n_node=n_node)
    model = trans_to_cuda(COTREC(global_neighbor = train_data.global_neighbor,n_node=n_node,opt = opt))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        file = open('/home/hzz/Gim/data/result.txt', "a")
        file.write('-------------------------------------------------------\n epoch: '+str(epoch)+'\n')
        file.close()
        model.select_num = 50#int(n_node/400)#int(n_node/(100*(min(epoch,1)+1)))
        metrics, total_loss = train_test(model, train_data, test_data, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        file = open('/home/hzz/Gim/data/result.txt', "a")
        file.write(' '+str(metrics)+'\n')
        file.close()
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            result = 'train_loss:\t{:.2f}\tRecall@{: >2d}: {:.2f}\tMRR{: >2d}: {:.2f}\tEpoch: {},  {}'.format(
                  total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1])
            file = open('/home/hzz/Gim/data/result.txt', "a")
            file.write(result+'\n')
            file.close()


if __name__ == '__main__':
    main()
