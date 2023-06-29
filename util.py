from tkinter.tix import Tree
import numpy as np
from torch.utils.data import Dataset,DataLoader,TensorDataset
from scipy.sparse import coo_matrix, csr_matrix
from operator import itemgetter
import random
import matplotlib.pyplot as plt

def get_global_neighbor(all_train,n_node,global_neighbor_num):
    adj = {}
    for sess in all_train:
        for i,item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                first = sess[i]
                adj.setdefault(first,{})
                for j in range(i+1,min(len(sess),i+4)):
                    second = sess[j]
                    adj.setdefault(second,{})
                    adj[first].setdefault(second,0)
                    adj[second].setdefault(first,0)
                    adj[first][second]+=1
                    adj[second][first]+=1
    global_neighbor = []
    global_neighbor.append(global_neighbor_num*[0])
    for i in range(1,1+n_node):
        if(adj.get(i,-1)==-1):
            global_neighbor.append((global_neighbor_num)*[0])
            continue
        sort_dict = sorted(adj[i].items(),key=lambda x:x[1],reverse=True)
        sort_key = [i[0] for i in sort_dict][:global_neighbor_num]
        if(len(sort_key)>global_neighbor_num-1):
            random.shuffle(sort_key)
            global_neighbor.append([i]+list(sort_key[:global_neighbor_num-1]))
        else:
            global_neighbor.append([i]+list(sort_key)+(global_neighbor_num-len(sort_key)-1)*[0])
    return global_neighbor    
class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0],dtype=object)
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)#样本长度
        self.shuffle = shuffle
        self.local_neighbor_num = 3
        self.global_neighbor_num = 100
        self.global_neighbor = get_global_neighbor(all_train,n_node,self.global_neighbor_num)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices
    
    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        self.filter = 20
        for session in inp:
            if(len(session)>self.filter):
                session = session[len(session)-self.filter:]
            num_node.append(len(np.nonzero(session)[0]))#每个session 非零长度
        max_n_node = np.max(num_node)
        session_len = []#每个session的原始长度
        reversed_sess_item = []#翻转后的填充session
        mask = []#[batch,max_n_node]
        local_neighbor ,local_2_neighbor = [],[]#[batch,len,nei]
        for session in inp:
            if(len(session)>self.filter):
                session = session[len(session)-self.filter:]
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
            neighbor_l ,local_neighbor_list,global_neighbor_list= {},[],[]
            reversed_session = list(reversed(session))
            for i,item in enumerate(reversed_session):
                neighbor_l.setdefault(item,[])
                if(i-1>=0):
                    neighbor_l[item].append(reversed_session[i-1])
                if(i+1<len(reversed_session)):
                    neighbor_l[item].append(reversed_session[i+1])
                # if(i-2>=0):
                #     neighbor_l[item].append(reversed_session[i-2])
                # if(i+2<len(reversed_session)):
                #     neighbor_l[item].append(reversed_session[i+2])
            neighbor_2l = {}
            for i,item in enumerate(reversed_session):
                if(len(neighbor_l[item])>self.local_neighbor_num-1):
                    random.shuffle(neighbor_l[item])
                    neighbor_item = [item]+neighbor_l[item][:self.local_neighbor_num-1]
                else:
                    neighbor_item = [item]+neighbor_l[item]+(self.local_neighbor_num-1-len(neighbor_l[item]))*[0]
                local_neighbor_list.append(neighbor_item)
                neighbor_2l.setdefault(item,neighbor_item)
            neighbor_2l.setdefault(0,self.local_neighbor_num*[0])
            for i in range(max_n_node-len(nonzero_elems)):
                local_neighbor_list.append(self.local_neighbor_num*[0])
            local_neighbor.append(local_neighbor_list)
            local_neighbor_2_list = []
            for i,item in enumerate(local_neighbor_list):
                neighbor_neighbor_list = []
                for neighbor in item:
                    neighbor_neighbor_list.append(neighbor_2l[neighbor])
                local_neighbor_2_list.append(neighbor_neighbor_list)
            local_2_neighbor.append(local_neighbor_2_list)
        # print(np.array(local_neighbor).shape)
        # print(np.array(local_2_neighbor).shape)
        # print(reversed_sess_item[0])
        # print(local_neighbor[0])
        # print(local_2_neighbor[0])
        # input()
        return self.targets[index]-1, session_len, items , reversed_sess_item, mask,local_neighbor,local_2_neighbor
