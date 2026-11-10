import json
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss, MSELoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.util import StandardScaler2
from models.STR_TTE import STR_TTE

veh_class = {'1':0, '2':1, '3':2, '4':3, "11":4, "12":5, "13":6, "14":7, "15":8, "16":9, "21":10, "22":11, "23":12,"24":13,"25":14,"26":15}


def STR_TTE_collate_func(data, args, info_all):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all

    time = torch.Tensor([d[-2] for d in data]) 
    linkids = []
    dateinfo = []
    vehinfo = []

    inds = []
    for ind, l in enumerate(data):
        linkids.append(np.asarray(l[1])) 
        vehinfo.append(l[-1])

        dateinfo.append(l[2:5]) 
        inds.append(l[0]) 
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16) 

    def info(xs, date, vehclass):
        infos = []
        length = 0
        for x in xs:
            info = edgeinfo[x] 
            infot = []

            infot.append(veh_class[vehclass])
            infot.append(info[0])
            length += info[0]
            infot.append(length)
            infot += list(date) 
            try:
                infot += [nodeinfo[info[1]][0],nodeinfo[info[1]][1],nodeinfo[info[2]][0],nodeinfo[info[2]][1]]
            except:
                print(info)

            infos.append(np.asarray(infot))


        return infos

    con_links = np.concatenate([info(b, dateinfo[ind], vehinfo[ind]) for ind, b in enumerate(linkids)], dtype='object')
    mask = np.arange(lens.max()) < lens[:, None]
   
    padded = np.zeros((*mask.shape, 1+2+3+4), dtype=np.float32) 
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 6:10] = scaler2.transform(con_links[:, 6:10]) 
    padded[mask] = con_links 
    rawlinks = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    rawlinks[mask] = np.concatenate(linkids) 

    def random_mask(tokens: np.array, rate: float):
        replaces = np.where(np.random.random(len(tokens)) <= rate)[0] 
        labels = np.full(len(tokens),dtype=np.int16, fill_value=-100) 
        tokens = tokens.copy()

        labels[replaces] = tokens[replaces] 
        tokens[replaces] = np.asarray([args.data_config['edges'] + 1] * len(replaces))   
        return labels, tokens


    mask_label_tmp = []
    sub_input_tmp = []
    for k in linkids:
        tmp1, tmp2 = random_mask(k, rate=args.mask_rate) 
        mask_label_tmp.append(tmp1) 
        sub_input_tmp.append(tmp2)
    mask_label = np.full(mask.shape, dtype=np.int16, fill_value=-100)
    mask_label[mask] = np.concatenate(mask_label_tmp) 

    linkindex = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    linkindex[mask] = np.concatenate(sub_input_tmp)
    mask_encoder = np.zeros(mask.shape, dtype=np.int16)
    mask_encoder[mask] = np.concatenate([[1]*k for k in lens]) 
    return {'links':torch.FloatTensor(padded), 'lens':torch.LongTensor(lens), 'inds': inds, 'mask_label': torch.LongTensor(mask_label),
            "linkindex":torch.LongTensor(linkindex), 'rawlinks': torch.LongTensor(rawlinks),'encoder_attention_mask': torch.LongTensor(mask_encoder)}, time

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size 
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        elif isinstance(dataset[0][1], list):
            self.lengths = [len(d[1]) for d in dataset]  
        else:
            self.lengths = [d[0]['lens'] for d in dataset]
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size 

 
        for i in range(chunks): 
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size] 
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True) 
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices 

        batches = (self.count - 1 + self.batch_size) // self.batch_size 

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size 

def load_datadoct_pre(args):
    global info_all
    edgeinfo, nodeinfo, scaler, scaler2 = None, None, None, None
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config
    with open(os.path.join(args.absPath,args.data_config['edges_dir']), 'rb') as f:
        edgeinfo = pickle.load(f) 
    with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
        nodeinfo = pickle.load(f) 


    if "fujian" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0,0]])
        scaler.mean_ = [9725.39, 46405.67]
        scaler.scale_ = [7874.15, 40485.33] 
        scaler2 = StandardScaler()
        scaler2.fit([[0,0,0,0]])
        scaler2.mean_ = [117.303670, 24.882551, 117.302704, 24.882660] 
        scaler2.scale_ = [0.491537, 0.327611, 0.491053, 0.330279] 
    else:
        ValueError("Wrong Dataset Name")

    scaler.mean_ = np.asarray(scaler.mean_, dtype=np.float64)
    scaler.scale_ = np.asarray(scaler.scale_, dtype=np.float64)
    scaler2.mean_ = np.asarray(scaler2.mean_, dtype=np.float64)
    scaler2.scale_ = np.asarray(scaler2.scale_, dtype=np.float64)
    info_all = [edgeinfo, nodeinfo, scaler, scaler2]


class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

def load_datadict(args):
    data = {}
    loader = {}
    if args.mode == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val', 'test']

    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True) 
        data[phase] = tdata 
        print(data[phase].shape) 

        loader[phase] = DataLoader(Datadict(data[phase]), batch_sampler=BatchSampler(data[phase], args.data_config['batch_size']),
                                       collate_fn=lambda x: eval(args.data_config['collate_fn'])(x, args, info_all),
                                       pin_memory=True) 

    loader['test'] = DataLoader(Datadict(data['test']), batch_size=args.data_config['batch_size'],
                                    collate_fn=lambda x: eval(args.data_config['collate_fn' ])(x, args, info_all),
                                    shuffle=False, pin_memory=True)
    return loader.copy(), StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    args.model_config = model_config
    model_config['pad_token_id'] = args.data_config['edges'] + 1 
    if "STR_TTE" in args.model:
        return STR_TTE(**model_config)

def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
        
            mse = MSELoss(reduction='mean').forward(preds.view(-1), labels)
            return mse
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 0.1))
            return mape
    elif args.loss == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels))
            return mape
    elif args.loss == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            preds = torch.squeeze(preds, 1)
            smoothL1 = SmoothL1Loss(reduction='mean', beta = args.loss_val).forward(preds, labels)
            return smoothL1 

    else:
        raise ValueError("Unknown loss function.")
    return loss


