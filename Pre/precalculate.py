"""
Precalcule statistic of dataset mean and std
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import json
from Pre.constants import RES_DIR
import numpy as np
import torch as th
import torch.utils.data

from Pre.utils import loadTestLabels
from Pre.utils import JsonDatasetNIm as JsonDataset
from tqdm import tqdm
from torchvision import transforms

def main(train_folder, batchsize=16, seed=42, cuda=True, outputf = ''):

    #if ONE_IMG_ONLY or model_type=='CNN_LSTM':


    print('has cuda?', cuda)
    # check whether val folder and train folder are the same

    XX = 54
    YY = 96

    # Will be changed to separe plus efective
    data = loadTestLabels(train_folder, 1, 1, 250)


    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)

    # Retrieve number of samples per set
    n_data = len(data)

    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loader
    data_loader = th.utils.data.DataLoader(
                                            JsonDataset(data,
                                                        preprocess=False,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        sequence=True,
                                                        pred_time = 1,
                                                        frame_interval = 12,
                                                        N_im = 1),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)


    numChannel = data_loader.dataset.numChannel

    print("Starting calcus...")
    # We iterate over epochs:
    test1 = th.zeros(numChannel, XX, YY, dtype = th.float32 )
    test2 =  th.zeros(2, dtype = th.float32 )



    mean = th.zeros(numChannel)
    std = th.zeros(numChannel)
    if cuda:
        mean, std = mean.cuda(), std.cuda()

    indx = 0
    for i, (inputs, targets) in enumerate(data_loader):

        # if we have incomplete mini_time_series we will skip it
        if( th.all(th.eq(inputs[0], test1))  and  th.all(th.eq(targets[0], test2))):
            continue

        if cuda:
            inputs = inputs.cuda()

        mean += th.mean(inputs, (0,2,3))
        std += th.std(inputs, (0,2,3))
        indx += 1

        if (indx%10 == 0):
            print('process :{:.4f}%'.format((indx*batchsize)/n_data))

    mean /= indx
    std /= indx

    print("  mean:\t\t", mean)
    print("  std:\t\t", std)

    stat = {}
    stat['mean'] = list(np.array(mean.cpu().numpy(), dtype=float))
    stat['std'] =   list(np.array(std.cpu().numpy(), dtype=float))

    stat_name = RES_DIR+outputf+'.json'
    json.dump(stat, open(stat_name,'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=32, type=int)
    parser.add_argument('--no-cuda', default=False, help='Disables CUDA training', type= bool)
    parser.add_argument('-o', '--output_file', default='stat_dataset_4_250episodes', help='output file name')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()

    main(train_folder=args.train_folder, batchsize=args.batchsize, cuda=args.cuda, outputf=args.output_file  )
