import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np
import pandas as pd
from train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.metric import calculate_metrics
from utils.util import to_var
import time


def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    weekID = []
    minuteID = []
    travel_lengths = []
    mean = [9725.39, 46405.67]
    scale = [7874.15, 40485.33]
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        links_len = len(features['links'])

        max_iter = min(64, links_len)

        for i in range(max_iter):
            try:

                value_1 = int(features['links'][i][0][3])
                value_2 = int(features['links'][i][0][5])
                weekID.append(value_1)
                minuteID.append(value_2)


                links_i = features['links'][i]  
                col2 = links_i[:, 1].cpu().detach().numpy()


                total_travel_length = 0.0


                for val in col2:

                    if np.isclose(val, 0, atol=1e-6):
                        normalized_val = 0.0
                    else:
                        normalized_val = val

                    if normalized_val != 0:
                        original_val = normalized_val * scale[0] + mean[0]
                    else:
                        original_val = 0

                    total_travel_length += original_val


                travel_lengths.append(total_travel_length)
            except (IndexError, ValueError) as e:
   
                print(f"警告：索引 {i} 提取失败，原因：{e}，已跳过")


        outputs, _ = model(features, args)

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    weekID = np.array(weekID)
    minuteID = np.array(minuteID)
    length = np.array(travel_lengths)
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([weekID, minuteID, length, pre2, inds]))
    np.save(os.path.join(args.model_folder, "targer.npy"), np.asarray([weekID, minuteID, length, tar2, inds]))



def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    load_datadoct_pre(args)
    data_loaders, scaler = load_datadict(args) 

    args.scaler = scaler 
    model = create_model(args)
    loss_func = create_loss(args)

    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    model = model.to(args.device)
    print(f'loss function: {args.loss}')
    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    print(model)
    # 训练
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)

        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError()

        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args)
    elif args.mode == 'resume':
        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()
        final_model = torch.load(os.path.join(model_folder, 'final_model.pkl'), map_location=args.device)
        start_epoch = final_model['epoch']
        model.load_state_dict(final_model['model_state_dict'], strict=False)
        optimizer.load_state_dict(final_model['optimizer_state_dict'])
        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args, start_epoch=start_epoch)
    print(args.loss)
    print(args.model_config)
    print(args.data_config)
    print(model_folder)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['model_state_dict'], strict=False)
    test_model(model, data_loaders['test'], args)

