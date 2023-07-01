import datetime
from glob import glob
import math
from torch.utils.data import Dataset,DataLoader,TensorDataset
from platform import node
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter,utils, GRU, MultiheadAttention
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo_matrix
import time
import random
from tqdm import tqdm
from numba import jit

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        output = neighbor_vector
        return output
class MultigrainTargetAttention(nn.Module):
    def __init__(self,layer,order,dim,dropout,use_tar_bias=False):
        super(MultigrainTargetAttention,self).__init__()
        self.dropout = dropout
        self.layer = layer
        self.order = order
        self.dim = dim
        self.use_tar_bias = use_tar_bias
        self.w_k = 10
        self.pos_embedding = nn.Embedding(20, self.dim)
        self.softmax,self.softmax_1 = nn.Softmax(dim=-1),nn.Softmax(dim=-2)
        self.tar_score = nn.Linear(self.dim, 1,bias = False)
        self.q = nn.ModuleList([nn.Linear(self.dim, 1,bias = False) for i in range(3*self.order)])
        self.glu = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(3*self.order)])
        self.glu2 = nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False)for i in range(3*self.order)])
        self.w = nn.ModuleList([nn.Linear(2*self.dim, self.dim, bias=False) for i in range(3*self.order)])
        self.glu_tar = nn.ModuleList([nn.Linear(self.dim, self.dim,bias=False) for i in range(3*self.order)])
        self.sess_Dropout = nn.Dropout(self.dropout)
        self.tar_Dropout = nn.Dropout(self.dropout)
    def multigrain_matching(self,sess_order_1,tar_order_1,sess_order_2,tar_order_2,sess_order_3,tar_order_3):
        sess_order_1 = torch.sum(sess_order_1,0)
        sess_order_2 = torch.sum(sess_order_2,0)
        sess_order_3 = torch.sum(sess_order_3,0)
        if(self.use_tar_bias):
            tar_order_1 = torch.sum(tar_order_1,0)
            tar_order_2 = torch.sum(tar_order_2,0)
            tar_order_3 = torch.sum(tar_order_3,0)
        else:
            tar_order_1 = tar_order_1[0]
            tar_order_2 = tar_order_1
            tar_order_3 = tar_order_1
        empty_tensor = torch.tensor([])
        score_item = torch.tensor([])
        if(sess_order_1.shape!=empty_tensor.shape):
            sess_order_1 = self.w_k*F.normalize(sess_order_1,dim=-1,p=2)
            tar_order_1 = F.normalize(tar_order_1,dim=-1,p=2)
            score_item = torch.sum(torch.mul(sess_order_1,tar_order_1),axis=-1)
        if(sess_order_2.shape!=empty_tensor.shape):
            sess_order_2 = F.normalize(sess_order_2,dim=-1,p=2)
            tar_order_2 = F.normalize(tar_order_2,dim=-1,p=2)
            score_item = score_item+torch.sum(torch.mul(sess_order_2,tar_order_2),axis=-1)
        if(sess_order_3.shape!=empty_tensor.shape):
            sess_order_3 = F.normalize(sess_order_3,dim=-1,p=2)
            tar_order_3 = F.normalize(tar_order_3,dim=-1,p=2)
            score_item = score_item+torch.sum(torch.mul(sess_order_3,tar_order_3),axis=-1)
        # print(score_item[0])
        return score_item
    def Generate_Multi_session(self,item_emb_interaction_list,target_emb_interaction_list,mask,pos = False,order_list=None):
        # item_emb_interaction:list((batch,k,len,dim)) target_emb_interaction:list((batch,k,dim)),session:list((batch,1)) masklist((batch,len))
        batch_size,select_num = item_emb_interaction_list[0].shape[0],item_emb_interaction_list[0].shape[1]
        sess_emb_total,tar_emb_total,sess_emb_total_2,tar_emb_total_2,sess_emb_total_3,tar_emb_total_3= torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([])
        length = mask.shape[1]
        mask = mask.float().unsqueeze(-1).unsqueeze(1)
        pos_emb = self.pos_embedding.weight[:length]#[batch_max_session_len,dim]
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(1).repeat(batch_size,select_num,1,1)#[batch,batch_max_sess_len,dim]
        for i in range(item_emb_interaction_list.shape[0]):
            item_emb_interaction,target_emb_interaction = item_emb_interaction_list[i],target_emb_interaction_list[i]
            if(pos):
                sess_item_pos = pos_emb+item_emb_interaction
                beta_sess_self = self.q[i](torch.sigmoid(self.glu[i](sess_item_pos)+self.glu2[i](item_emb_interaction*torch.unsqueeze(target_emb_interaction,-2))))#[batch,k,len,1]
                # beta_sess_self = self.q[i](torch.sigmoid(self.glu[i](sess_item_pos)))#[batch,k,len,1]
            else:
                beta_sess_self = self.q[i](torch.sigmoid(self.glu[i](item_emb_interaction)+self.glu2[i](item_emb_interaction*torch.unsqueeze(target_emb_interaction,-2))))#[batch,k,len,1]
                # beta_sess_self = self.q[i](torch.sigmoid(self.glu[i](item_emb_interaction)))#[batch,k,len,1]
            
            # beta_tar = torch.matmul(item_emb_interaction,torch.unsqueeze(self.glu_tar[i](target_emb_interaction),-1))#[batch,k,len,1]
            
            # beta = self.softmax_1(beta_sess_self*mask+beta_tar*mask)
            beta = self.softmax_1(beta_sess_self*mask)
            sess_emb_interaction = torch.sum(item_emb_interaction*beta,-2)
            if(i==0):
                sess_emb_total = sess_emb_interaction.unsqueeze(0)
                tar_emb_total = target_emb_interaction.unsqueeze(0)
            else:
                if(order_list[i]==1):
                    sess_emb_total = torch.cat([sess_emb_total,sess_emb_interaction.unsqueeze(0)],0)
                    tar_emb_total = torch.cat([tar_emb_total,target_emb_interaction.unsqueeze(0)],0)
                # elif(order_list[i]==2):
                #     empty_tensor = torch.tensor([])
                #     if(sess_emb_total_2.shape==empty_tensor.shape):
                #         sess_emb_total_2 = sess_emb_interaction.unsqueeze(0)
                #         tar_emb_total_2 = target_emb_interaction.unsqueeze(0)
                #     else:
                #         sess_emb_total_2 = torch.cat([sess_emb_total_2,sess_emb_interaction.unsqueeze(0)],0)
                #         tar_emb_total_2 = torch.cat([tar_emb_total_2,target_emb_interaction.unsqueeze(0)],0)
                # else:
                #     sess_emb_total_3 = sess_emb_interaction.unsqueeze(0)
                #     tar_emb_total_3 = target_emb_interaction.unsqueeze(0)
        # print('order=1: ',sess_emb_total.shape,'--',tar_emb_total.shape)
        # print('order=2: ',sess_emb_total_2.shape,'--',tar_emb_total_2.shape)
        # print('order=3: ',sess_emb_total_3.shape,'--',tar_emb_total_3.shape)
        # input()
        return self.multigrain_matching(sess_emb_total,tar_emb_total,sess_emb_total_2,tar_emb_total_2,sess_emb_total_3,tar_emb_total_3)
        if(self.use_tar_bias):
            return torch.sum(sess_emb_total,0),torch.sum(tar_emb_total,0)#1
        else:
            return torch.sum(sess_emb_total,0),target_emb_interaction_list[0]#2
    def forward(self, sess_embedding_layer,target_embedding_layer,mask,pos = False):
        sess_embedding_list,target_embedding_list = torch.tensor([]),torch.tensor([])
        for i in range(1,pow(2,self.layer+1)):
            now = i
            index_list = []
            for j in range(self.layer+1):
                if(now%2==1):
                    index_list.append(j)
                now=int(now/2)
            if(len(index_list)>self.order):
                continue
            index_tensor = trans_to_cuda(torch.tensor(index_list).long())
            muilti_sess_embedding = torch.index_select(sess_embedding_layer,dim = 0,index = index_tensor)
            muilti_sess_embedding = F.normalize(torch.sum(muilti_sess_embedding,dim=0),dim=-1,p=2)
            muilti_tar_embedding = torch.index_select(target_embedding_layer,dim=0,index = index_tensor)
            muilti_tar_embedding = F.normalize(torch.sum(muilti_tar_embedding,dim=0),dim=-1,p=2)
            if(sess_embedding_list.shape[0]==0):
                sess_embedding_list = muilti_sess_embedding.unsqueeze(0)
                target_embedding_list = muilti_tar_embedding.unsqueeze(0)
                order_list = torch.tensor([len(index_list)])
            else:
                sess_embedding_list = torch.cat([sess_embedding_list,muilti_sess_embedding.unsqueeze(0)],0)
                target_embedding_list = torch.cat([target_embedding_list,0.5*muilti_tar_embedding.unsqueeze(0)],0)
                order_list = torch.cat([order_list,torch.tensor([len(index_list)])])
        return self.Generate_Multi_session(sess_embedding_list,target_embedding_list,mask,pos,order_list)
class InteractionGNNAggregator(nn.Module):
    def __init__(self,dim,dropout,layer=0,name = None):
        super(InteractionGNNAggregator,self).__init__()
        self.dropout = dropout
        self.dim = dim 
        self.layer = layer
        self.softmax,self.softmax_1 = nn.Softmax(dim=-1),nn.Softmax(dim=-2)
        self.w_1,self.w_3,self.w_5,self.w_6 = nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(2*self.dim, self.dim,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)])
        self.w_2,self.w_4 = nn.ModuleList([nn.Linear(self.dim, 1,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(self.dim, 1,bias = False) for i in range(self.layer)])
        self.ignn_glu1,self.ignn_glu2 = nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)]) 
        self.gate_1,self.gate_2 = nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)]),nn.ModuleList([nn.Linear(self.dim, self.dim,bias = False) for i in range(self.layer)])
        self.MHSA_sess = nn.ModuleList([torch.nn.MultiheadAttention(embed_dim = self.dim,num_heads = 10,dropout = self.dropout) for i in range(self.layer)])
         
    def forward(self,sess_self_vector,sess_neighbor_vector,sess_neighbor2_vector,tar_self_vector,tar_neighbor_vector,session_len):
        batch_size,len,neighbor = sess_neighbor_vector.shape[0],sess_neighbor_vector.shape[1],sess_neighbor_vector.shape[2]
        select_num = tar_self_vector.shape[1]
        sess_embedding_layer = torch.unsqueeze(sess_self_vector,1).repeat(1,select_num,1,1).unsqueeze(0)
        target_embedding_layer = tar_self_vector.unsqueeze(0)
        for i in range(self.layer):
            # way1
            sess_neighbor_att = self.w_1[i](torch.mean(sess_embedding_layer[i],1).unsqueeze(-2)*sess_neighbor_vector)#batch,k,len,num,dim
            sess_neighbor_att = F.leaky_relu(sess_neighbor_att,negative_slope=0.2)
            sess_neighbor_att = self.w_2[i](sess_neighbor_att)
            sess_neighbor_att = self.softmax_1(sess_neighbor_att)
            sess_neighbor_total = torch.sum(sess_neighbor_att*sess_neighbor_vector,-2).unsqueeze(1).repeat(1,select_num,1,1)#batch,k,len,dim
            
            # # way0
            # sess_neighbor_att = self.w_1[i](sess_embedding_layer[i].unsqueeze(-2)*sess_neighbor_vector.unsqueeze(1))#batch,k,len,num,dim
            # sess_neighbor_att = self.w_5[i](target_embedding_layer[i].unsqueeze(-2).unsqueeze(-2)*sess_neighbor_vector.unsqueeze(1))#batch,k,len,num,dim#sess_neighbor_att+0.2*self.w_5[i](target_embedding_layer[i].unsqueeze(-2).unsqueeze(-2)+sess_neighbor_vector.unsqueeze(1))#batch,k,len,num,dim
            # sess_neighbor_att = F.leaky_relu(sess_neighbor_att,negative_slope=0.2)
            # sess_neighbor_att = self.w_2[i](sess_neighbor_att)
            # sess_neighbor_att = self.softmax_1(sess_neighbor_att)
            # sess_neighbor_total = torch.sum(sess_neighbor_att*sess_neighbor_vector.unsqueeze(1),-2)#batch,k,len,dim

            # sess_neighbor_gate = torch.matmul(self.gate_1[i](sess_embedding_layer[i]),self.gate_2[i](target_embedding_layer[0]).unsqueeze(-1))/math.sqrt(self.dim)
            # sess_item_embed_layeri = sess_neighbor_gate*sess_embedding_layer[i]+(1-sess_neighbor_gate)*sess_neighbor_total

            sess_item_embed_layeri = sess_neighbor_total
            sess_item_embed_layeri = F.normalize(sess_item_embed_layeri,dim=-1,p=2).unsqueeze(0)
            sess_embedding_layer = torch.cat([sess_embedding_layer,sess_item_embed_layeri],0)

            if(i<=0 and self.layer>1):
                #update sess_neighbor_vector
                sess_neighbor2_att = self.w_1[i](sess_neighbor_vector.unsqueeze(-2)*sess_neighbor2_vector)
                sess_neighbor2_att = F.leaky_relu(sess_neighbor2_att,negative_slope=0.2)
                sess_neighbor2_att = self.w_2[i](sess_neighbor2_att)
                sess_neighbor2_att = self.softmax_1(sess_neighbor2_att)
                sess_neighbor_vector = torch.sum(sess_neighbor2_att*sess_neighbor2_vector,dim=-2)
                # sess_neighbor_vector = F.dropout(sess_neighbor_vector, 0.5, training=self.training)
                # sess_neighbor_vector = F.normalize(sess_neighbor_vector,dim=-1,p=2)

            
            # #tar
            # sess_mean = torch.div(torch.sum(sess_embedding_layer[i-1],2),session_len.unsqueeze(1)).unsqueeze(-2)
            sess_last = sess_embedding_layer[i-1][:,:,:1,:]
            tar_neighbor_att = self.w_3[i](torch.cat([target_embedding_layer[i].unsqueeze(-2)*tar_neighbor_vector,sess_last+tar_neighbor_vector],-1))
            # tar_neighbor_att = self.w_3[i](target_embedding_layer[i].unsqueeze(-2)*tar_neighbor_vector)#self
            # tar_neighbor_att = self.w_3[i](sess_mean+tar_neighbor_vector)#interaction
            # tar_neighbor_att = F.leaky_relu(tar_neighbor_att,negative_slope=0.2)
            tar_neighbor_att = torch.tanh(tar_neighbor_att)
            tar_neighbor_att = self.w_4[i](tar_neighbor_att)
            tar_neighbor_att = self.softmax_1(tar_neighbor_att)
            tar_neighbor_total = torch.sum(tar_neighbor_att*tar_neighbor_vector,-2)#(batch,k,dim)
            # tar_neighbor_total = F.dropout(tar_neighbor_total, 0.2, training=self.training)
            target_embedding_layeri = (tar_neighbor_total+target_embedding_layer[0])/2
            target_embedding_layeri = F.normalize(target_embedding_layeri,dim=-1,p=2).unsqueeze(0)
            target_embedding_layer = torch.cat([target_embedding_layer,target_embedding_layeri],0)
            target_embedding_layer = torch.cat([target_embedding_layer,target_embedding_layer],0)
        return sess_embedding_layer,target_embedding_layer

class COTREC(Module):
    def __init__(self, global_neighbor, n_node, opt):
        super(COTREC, self).__init__()
        self.emb_size = opt.embSize
        self.batch_size = opt.batchSize
        self.n_node = n_node
        self.dataset = opt.dataset
        self.L2 = opt.l2
        self.lr = opt.lr
        self.layers = int(opt.layer)
        self.beta = opt.beta
        self.lam = opt.lam
        self.order = opt.order
        self.dropout = opt.dropout
        self.use_tar_bias = opt.use_tar_bias
        self.w_k = 10
        self.select_num=int(self.n_node/100)
        self.neighbor_num = 3
        self.global_aggregator = GlobalAggregator(self.emb_size, self.dropout, act=torch.relu)
        self.InteractionGNN_aggregator = InteractionGNNAggregator(self.emb_size,self.dropout,self.layers)
        self.multigraintarget_att = MultigrainTargetAttention(dim=self.emb_size,dropout=self.dropout,layer=self.layers,order=self.order,use_tar_bias = self.use_tar_bias)
        self.global_neighbor = trans_to_cuda(torch.tensor(np.array(global_neighbor)[:,:self.neighbor_num]).long())
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200 #max_length
        if self.dataset == 'RetailRocket':
            self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.w1_select = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w3_select = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.w2_select = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1_select = nn.Linear(self.emb_size, self.emb_size)
        self.glu2_select = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.mhsa = nn.MultiheadAttention(self.emb_size, 10)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            # weight.data.normal_(0,0.1)
    def generate_interaction_target(self,select_index, item_embedding,reversed_sess_item,local_neighbor,local_2_neighbor,session_len):
        # input :: select_index:(batch,select_num), item_embedding:(node,dim), session_len:(batch,1), reversed_sess_item:(batch,len), mask:(batch,len) local_neighbor:batch,len,n_sum
        # output :: sess_item_embed:(batch,len,dim), target_embedding:(batch,select_num,dim)
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        len,k  = list(reversed_sess_item.shape)[1],select_index.shape[1]
        target_embedding = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(select_index,(-1,))),(self.batch_size,k,self.emb_size))#[batch,len,dim]
        item_embedding = torch.cat([zeros, item_embedding], 0)#[node,dim]
        sess_item_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(reversed_sess_item,(-1,))),(self.batch_size,len,self.emb_size))#[batch,len,dim]
        #global-ignn
        tar_neighbor = torch.reshape(torch.index_select(self.global_neighbor,0,torch.reshape(select_index+1,(-1,))),(self.batch_size,self.select_num,self.neighbor_num))#[batch,k,g_num]
        # print(tar_neighbor[0,0,:])
        # idx = torch.randperm(tar_neighbor.shape[-1])
        # tar_neighbor = tar_neighbor[:,:,idx].view(tar_neighbor.size())[:,:,:10]
        # print(tar_neighbor[0,0,:])
        # input()
        tar_neighbor_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(tar_neighbor,(-1,))),(self.batch_size,self.select_num,-1,self.emb_size))#[batch,k,g_num,dim]   
        
        # sess_neighbor = torch.reshape(torch.index_select(self.global_neighbor,0,torch.reshape(reversed_sess_item,(-1,))),(self.batch_size,len,self.neighbor_num))#[batch,len,g_num]
        # sess_neighbor_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(sess_neighbor,(-1,))),(self.batch_size,len,self.neighbor_num,self.emb_size))#[batch,len,g_num,dim]
        
        #local-ignn
        neighbor_num = local_neighbor.shape[2]
        sess_neighbor_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(local_neighbor,(-1,))),(self.batch_size,len,neighbor_num,self.emb_size))#[batch,len,l_num,dim]
        sess_neighbor2_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(local_2_neighbor,(-1,))),(self.batch_size,len,neighbor_num,neighbor_num,self.emb_size))#[batch,len,l_num,l_num,dim]

        
        sess_item_embed = F.normalize(sess_item_embed,dim=-1,p=2)
        sess_neighbor_embed = F.normalize(sess_neighbor_embed,dim=-1,p=2)
        sess_neighbor2_embed = F.normalize(sess_neighbor2_embed,dim=-1,p=2)
        target_embedding = F.normalize(target_embedding,dim=-1,p=2)
        tar_neighbor_embed = F.normalize(tar_neighbor_embed,dim=-1,p=2)
        sess_embedding_layer,target_embedding_layer = self.InteractionGNN_aggregator(sess_item_embed,sess_neighbor_embed,sess_neighbor2_embed,target_embedding,tar_neighbor_embed,session_len)
        return sess_embedding_layer,target_embedding_layer
    def select_topk(self, item_embedding, reversed_sess_item, session_len, mask,pos = False):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        len  = list(reversed_sess_item.shape)[1]
        sess_item_embed = torch.cuda.FloatTensor(self.batch_size, len, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)#[node,dim]
        get_sess_item = lambda i: item_embedding[reversed_sess_item[i]]
        for i in torch.arange(self.batch_size):#batch_size
            sess_item_embed[i] = get_sess_item(i)#[batch,len,dim]
        if(pos == True):
            pos_emb = self.pos_embedding.weight[:len]#[batch_max_session_len,dim]
            beta_sess_self = pos_emb+sess_item_embed
        else:
            beta_sess_self = sess_item_embed
        mask = mask.float().unsqueeze(-1)#[batch,len,1]
        sess_mean = torch.unsqueeze(torch.div(torch.sum(sess_item_embed,1),session_len),1).repeat(1,len,1)#[batch,1,dim]
        beta_sess = torch.sigmoid(self.glu1_select(beta_sess_self) + self.glu2_select(sess_mean))
        beta_sess = torch.matmul(beta_sess, self.w2_select)#[batch,len,1]
        beta = beta_sess * mask#[batch,len,1]
        softatt_sess = torch.sum(beta * sess_item_embed, 1)

        softatt_sess = self.w_k * F.normalize(softatt_sess, dim=-1, p=2)#[batch,dim]
        item_embedding = F.normalize(item_embedding, dim=-1, p=2)
        score = torch.mm(softatt_sess,torch.transpose(item_embedding[1:],1,0))
        value, index = score.topk(self.select_num, dim=1, largest=True, sorted=True)#[batch,topk]
        return value,index,score
    def loss_target(self,score,index,tar):
        score_all = torch.cuda.FloatTensor(self.batch_size, self.n_node).fill_(0)
        for i in torch.arange(score_all.shape[0]):
            score_all[i][index[i]] = score[i]
        loss = self.loss_function(score_all,tar)
        return loss,score_all
    def forward(self, session_item, session_len, reversed_sess_item, mask , tar, local_neighbor,local_2_neighbor):
        #local-gnn
        # batch_size,len,neighbor_num= reversed_sess_item.shape[0],reversed_sess_item.shape[1],local_neighbor.shape[2]
        # zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # item_embedding = torch.cat([zeros,self.embedding.weight],0)
        # sess_neighbor_weight = torch.cuda.FloatTensor(batch_size,len,neighbor_num).fill_(0)#[batch,len,g_num]
        # sess_neighbor_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(local_neighbor,(-1,))),(batch_size,len,neighbor_num,self.emb_size))#[batch,len,g_num,dim]
        # sess_self_embed = torch.reshape(torch.index_select(item_embedding,0,torch.reshape(reversed_sess_item,(-1,))),(batch_size,len,self.emb_size))#[batch,len,dim]
        # sess_mean = torch.div(torch.sum(sess_self_embed, 1),session_len).unsqueeze(1).repeat(1,len,1)#[batch,dim]
        # item_embeddings_i = self.global_aggregator(self_vectors=sess_self_embed,
        #                             neighbor_vector=sess_neighbor_embed,
        #                             masks=None,
        #                             batch_size=batch_size,
        #                             neighbor_weight=sess_neighbor_weight,
        #                             extra_vector=sess_self_embed)             
        # item_embeddings_i = F.normalize(item_embeddings_i,dim=-1,p=2)
        if self.dataset == 'Tmall':
            # for Tmall dataset, we do not use position embedding to learn temporal order
            value,index,score = self.select_topk(self.embedding.weight, reversed_sess_item, session_len, mask,pos=False)#(3)
            loss_select = self.loss_function(score, tar)
            item_emb_interaction,target_emb_interaction = self.generate_interaction_target(index,self.embedding.weight, reversed_sess_item,local_neighbor,local_2_neighbor,session_len)
            item_emb_interaction = F.normalize(item_emb_interaction, dim=-1, p=2)
            target_emb_interaction = F.normalize(target_emb_interaction, dim=-1, p=2)
            # sess_emb_i,target_emb = self.multigraintarget_att(item_emb_interaction,target_emb_interaction,mask=mask,pos=False)
            scores_item = self.multigraintarget_att(item_emb_interaction,target_emb_interaction,mask=mask,pos=False)
        else:
            value,index,score = self.select_topk(self.embedding.weight, reversed_sess_item, session_len, mask,pos=True)#(3)
            loss_select = self.loss_function(score, tar)
            item_emb_interaction,target_emb_interaction = self.generate_interaction_target(index,self.embedding.weight, reversed_sess_item,local_neighbor,local_2_neighbor,session_len)
            item_emb_interaction = F.normalize(item_emb_interaction, dim=-1, p=2)
            target_emb_interaction = F.normalize(target_emb_interaction, dim=-1, p=2)
            # sess_emb_i,target_emb = self.multigraintarget_att(item_emb_interaction,target_emb_interaction,mask=mask,pos=True)
            scores_item = self.multigraintarget_att(item_emb_interaction,target_emb_interaction,mask=mask,pos=True)
        # sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)#[batch,k,dim]
        # target_emb = F.normalize(target_emb, dim=-1, p=2)#[batch,k,dim] (3)
        # scores_item = torch.sum(torch.mul(sess_emb_i,target_emb),axis=-1)#(3)
        loss_item, scores_item = self.loss_target(scores_item, index, tar)#(3)
        return self.lam*loss_select,self.beta*loss_item, scores_item, self.lam


def forward(model, i, data):#get slice；as data input；get loss
    tar, session_len, session_item, reversed_sess_item, mask ,local_neighbor,local_2_neighbor= data.get_slice(i)#diff_mask=[100,node]
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    local_neighbor = trans_to_cuda(torch.Tensor(local_neighbor).long())
    local_2_neighbor = trans_to_cuda(torch.Tensor(local_2_neighbor).long())
    loss_select, loss_item, scores_item, loss_diff = model(session_item, session_len, reversed_sess_item, mask,tar,local_neighbor,local_2_neighbor)
    return tar, scores_item, loss_select, loss_item, loss_diff
def get_slice(batch):
    items, num_node = [], []
    targets = []
    for session,tar in batch:
        num_node.append(len(np.nonzero(session)[0]))#每个session 非零长度
        targets.append(tar-1)
    max_n_node = np.max(num_node)
    session_len = []#每个session的原始长度
    reversed_sess_item = []#翻转后的填充session
    mask = []#[batch,max_n_node]
    for session,tar in batch:
        nonzero_elems = np.nonzero(session)[0]
        session_len.append([len(nonzero_elems)])
        items.append(session + (max_n_node - len(nonzero_elems)) * [0])
        mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
        reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
    return targets, session_len, items , reversed_sess_item, mask
def train_test(model, train_data, test_data, epoch):
    file = open('/home/hzz/Gim/data/result.txt', "a")
    print('start training: ', datetime.datetime.now())
    file.write('start training: '+ str(datetime.datetime.now())+'\n')
    file.close()
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    # slices = slices[:int(0.1*len(slices))]
    for i in tqdm(slices,total=len(slices)):
        model.zero_grad()
        tar, scores_item, loss_select, loss_item, loss_diff = forward(model, i, train_data)
        loss = loss_select+loss_item
        loss.backward()
        for name,p in model.named_parameters():
            if  p.requires_grad and torch.is_tensor(p.grad):
                utils.clip_grad.clip_grad_norm_(p,10)
        model.optimizer.step()
        total_loss += loss.item()
    model.scheduler.step()
    print('---',model.optimizer.param_groups[0]['lr'])
    file = open('/home/hzz/Gim/data/result.txt', "a")
    print('\tLoss:\t%.3f' % total_loss)
    file.write('Loss = '+ str(total_loss)+' \n')
    file.close()
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    file = open('/home/hzz/Gim/data/result.txt', "a")
    file.write('start training: '+ str(datetime.datetime.now())+'\n')
    file.close()
    model.eval()
    # model.global_neighbor = trans_to_cuda(torch.tensor(np.array(test_data.global_neighbor)[:,:model.neighbor_num]).long())
    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices,total=len(slices)):
        tar,scores_item, con_loss, loss_item, loss_diff = forward(model, i, test_data)
        index = scores_item.topk(20)[1]
        index = trans_to_cpu(index).detach().numpy()
        # scores = trans_to_cpu(scores_item).detach().numpy()
        # index = []
        # for idd in range(model.batch_size):
        #     index.append(find_k_largest(top_K[-1], scores[idd]))
        # index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


