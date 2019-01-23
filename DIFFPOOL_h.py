import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GraphConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import ToDense
from earlystopping import Earlystop
import random
import argparse
import pickle
import os
import csv
from torch.utils.data.dataset import Subset
import numpy as np
import time
def gmp(tensor):
    return torch.max(tensor,dim=1)[0]

def gap(tensor):
    return torch.mean(tensor,dim=1)



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='Fixed seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='Keeping node ratio')
parser.add_argument('--dataname', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/Letter-high(med,low)/TWITTER-Real-Graph-Partial/COIL-DEL/COIL-RAG')
parser.add_argument('--mode', type=str, default='search',
                    help='search/seed')



args = parser.parse_args()
print(args)


def delete_edge_attr(data):
    data.edge_attr = None
    return data




random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def my_transform(data):
    data.x = F.normalize(data.x,p=2,dim=-1)
    return data

if not os.path.exists(os.path.join('./data',args.dataname)):
    os.mkdir(os.path.join('./data',args.dataname))

data_seed_dir = os.path.join('./data',args.dataname,str(args.seed)+'.pkl')

if not os.path.exists(data_seed_dir):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataname)
    dataset = TUDataset(path, name= args.dataname)#pre_transform=my_transform,transform=T.ToDense(args.max_nodes),pre_filter=MyFilter())
    max_nodes = max([x.num_nodes for x in dataset])
     
    del dataset
    dataset = TUDataset(path, name= args.dataname,pre_transform=my_transform)#,transform=T.ToDense(max_nodes))
    #dataset.max_nodes = max_nodes
    dataset = dataset.shuffle()
    dataset = dataset.shuffle()
    with open(data_seed_dir, 'wb') as f:
        pickle.dump(dataset, f)
    print('Seed Data Saved : ',data_seed_dir)
else:
    with open(data_seed_dir, 'rb') as f:
        dataset = pickle.load(f)
    print('Seed Data Loaded : ',data_seed_dir)


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES_d')
# dataset = dataset = TUDataset(
#     path,
#     name='ENZYMES',
#     transform=T.ToDense(max_nodes),
#     pre_filter=MyFilter())

# dataset = dataset.shuffle()

class DenseGCN(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super(DenseGCN,self).__init__()
        self.W = torch.nn.Parameter(torch.randn(in_channels,out_channels,requires_grad=True))
        self.non_linearity = torch.nn.ReLU()
        #self.bias = nn.Parameter(torch.randn(1,out_channels,requires_grad=True))
    
    def forward(self,x,adj):
        
        adj = adj + torch.eye(adj.size(1),requires_grad=False).to(x.device).unsqueeze(0)

        degree = torch.sum(adj,dim=2)
        degree = 1/torch.sqrt(degree)
        #degree = torch.eye(adj.size(1),requires_grad=False).to(x.device).repeat(adj.size(0),1,1) * (1/torch.sqrt(degree))
        #laplacian = torch.matmul(torch.matmul(degree,adj),degree)
        laplacian = degree.unsqueeze(1)*adj*degree.unsqueeze(-1)
        x = torch.matmul(laplacian,x)
        return self.non_linearity(torch.matmul(x,self.W))#+self.bias)

def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (ByteTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    reg = adj - torch.matmul(s, s.transpose(1, 2))
    reg = torch.norm(reg, p=2)
    reg = reg / adj.numel()


    log = torch.log(s + 1e-14)
    entropy = s*log
    entropy_loss = -torch.sum(entropy)
    entropy_loss = entropy_loss/float(s.size(0))
    
    reg = reg + entropy_loss

    return out, out_adj, reg






class DIFFPOOL(torch.nn.Module):

    def __init__(self,in_channels,out_channels,num_nodes):
        super(DIFFPOOL,self).__init__()
        self.gnn_pool = DenseGCN(in_channels,num_nodes)
        self.gnn_embed = DenseGCN(in_channels,out_channels)
    def forward(self,x,adj,mask=None):
        s = F.relu(self.gnn_pool(x,adj))
        x = F.relu(self.gnn_embed(x,adj))
        x, adj, reg = dense_diff_pool(x, adj, s, mask)
        return x,adj,reg

class Net(torch.nn.Module):
    def __init__(self,max_nodes,nhid):
        super(Net, self).__init__()



        self.conv1 = DenseGCN(dataset.num_features, nhid)
        self.pool1 = DIFFPOOL(nhid, nhid,ceil(args.ratio*max_nodes))
        
        self.conv2 = DenseGCN(nhid, nhid)
        self.pool2 = DIFFPOOL(nhid, nhid,ceil(args.ratio**2*max_nodes))

        self.conv3 = DenseGCN(nhid, nhid)
        self.pool3 = DIFFPOOL(nhid, nhid,ceil(args.ratio**3*max_nodes))

        self.lin1 = torch.nn.Linear(nhid*2, nhid)
        self.lin2 = torch.nn.Linear(nhid, int(nhid/2))
        self.lin3 = torch.nn.Linear(int(nhid/2), dataset.num_classes)
        self.to_dense = ToDense()

    def forward(self,data):
        x,adj,mask = data.x, data.adj, data.mask
        
        x = F.relu(self.conv1(x,adj))

        x,adj,reg1 = self.pool1(x,adj,mask)
        
        x1 = torch.cat([gmp(x), gap(x)], dim=1)
        
        x = F.relu(self.conv2(x,adj))
        
        x,adj,reg2 = self.pool2(x,adj)
        x2 = torch.cat([gmp(x), gap(x)], dim=1)
        
        x = F.relu(self.conv3(x,adj))
        
        x,adj,reg3 = self.pool3(x,adj)
        
        x3 = torch.cat([gmp(x), gap(x)], dim=1)
        x = x1 + x2 + x3 
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        reg = reg1+reg2+reg3
        return F.log_softmax(x, dim=-1), reg





max_nodes = max([x.num_nodes for x in dataset])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(loader,max_nodes):
    model.train()
    
    loss_all = 0
    loader.dataset.transform = T.Compose([delete_edge_attr,T.ToDense(max_nodes[0])])
    i = 0
    for data in loader:
        data = data.to(device)
        data.y = data.y.squeeze(-1)
        optimizer.zero_grad()
        output,reg = model(data)
       
        loss = F.nll_loss(output, data.y) + reg
        loss.backward()
        loss_all += data.x.size(0) * loss.item()
        optimizer.step()
        i+=1
        loader.dataset.transform = T.Compose([delete_edge_attr,T.ToDense(max_nodes[i])])
    
    return loss_all / len(loader.dataset)


def test(loader,max_nodes):
    model.eval()

    correct = 0
    loss_all = 0
    loader.dataset.transform = T.Compose([delete_edge_attr,T.ToDense(max_nodes[0])])
    i=0
    for data in loader:
        data.y = data.y.squeeze(-1)
        data = data.to(device)
        output = model(data)[0]
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss = F.nll_loss(output, data.y)
        loss_all += data.x.size(0)* loss.item()
        i+=1
        loader.dataset.transform = T.Compose([delete_edge_attr,T.ToDense(max_nodes[i])])
        
    return correct / len(loader.dataset), loss_all / len(loader.dataset) 




# split data to 10-fold
fold_size = ceil(len(dataset)/10)
final_acc = 0.
best_acc = 0.
for fold in range(10):
    model = Net(max_nodes,args.nhid).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if fold != 9:
        test_index = range(fold_size*fold,fold_size*(fold+1))
    else:
        test_index = range(fold_size*fold,len(dataset))
    test_mask = np.zeros(len(dataset),dtype=np.bool)
    test_mask[test_index] = 1
    train_mask = np.invert(test_mask)
    valid_idx = int(len(dataset)*0.9*0.1)
    valid_idx = np.where(train_mask)[0][:valid_idx]
    valid_mask = np.zeros(len(dataset),dtype=np.bool)
    valid_mask[valid_idx] = 1
    train_mask = np.invert(test_mask + valid_mask)

    test_mask = torch.LongTensor(np.where(test_mask)[0])
    valid_mask = torch.LongTensor(np.where(valid_mask)[0])
    train_mask = torch.LongTensor(np.where(train_mask)[0])
    
    train_max_nodes = [ max([x.num_nodes for x in dataset[train_mask][i:i+args.batch_size]])   for i in range(0,len(dataset[train_mask]),args.batch_size) ]
    test_max_nodes = [ max([x.num_nodes for x in dataset[test_mask][i:i+args.batch_size]])   for i in range(0,len(dataset[test_mask]),args.batch_size) ]
    valid_max_nodes = [ max([x.num_nodes for x in dataset[valid_mask][i:i+args.batch_size]])   for i in range(0,len(dataset[valid_mask]),args.batch_size) ]
    
    train_max_nodes.append(max([x.num_nodes for x in dataset[train_mask][-args.batch_size-1:]]))
    test_max_nodes.append(max([x.num_nodes for x in dataset[test_mask][-args.batch_size-1:]]))
    valid_max_nodes.append(max([x.num_nodes for x in dataset[valid_mask][-args.batch_size-1:]]))

    test_loader = DenseDataLoader(dataset[test_mask], batch_size=args.batch_size,shuffle=False)
    valid_loader = DenseDataLoader(dataset[valid_mask], batch_size=args.batch_size,shuffle=False)
    train_loader = DenseDataLoader(dataset[train_mask], batch_size=args.batch_size,shuffle=False)

    fold_best_acc = 0.
    fold_val_loss = 100000000.
    fold_val_acc = 0.
    patience = 0
    
    for epoch in range(1, 100001):
        t = time.time()
        loss = train(train_loader,train_max_nodes)
        val_acc,val_loss = test(valid_loader,valid_max_nodes)
        train_acc,_ = test(train_loader,train_max_nodes)
        
        if fold_val_loss >= val_loss:
            fold_val_loss = val_loss

            test_acc,test_loss = test(test_loader,test_max_nodes)
            fold_best_acc = test_acc
        
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Valid Acc: {:.5f}, Test Acc: {:.5f}, Time: {:.2f}s'.
                format(epoch, loss, train_acc, val_acc, test_acc,time.time()-t))
            patience = 0
        elif patience >= 50:
            print("{:01d} Fold Early Stop Epoch: {:03d}".format(fold,epoch))
            break
        else:
            patience += 1
    best_acc += fold_best_acc

    del model
    del optimizer
print('Final Accuracy : ', best_acc/10)

f = open(os.path.join('./data',args.dataname,args.mode+'_diffpool_result.csv'), 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([best_acc/10,args.seed,args.lr,args.nhid,args.weight_decay,args.batch_size,args.ratio])
f.close()