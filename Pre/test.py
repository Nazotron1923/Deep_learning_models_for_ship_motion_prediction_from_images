"""
predict dataset
"""
from __future__ import print_function, division, absolute_import
import json
import argparse
import time
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RES_DIR, ONE_IMG_ONLY, DATASET_SEQUENCE
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from Pre.utils import loadLabels, loadTestLabels, loadTrainLabels
from Pre.models import ConvolutionalNetwork, CNN_LSTM
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc
import os

def main(test_folder, batchsize=64,
         seed=42, cuda=True, random_trans=0.0,
         model_type="cnn", weights='', time_gap=5, output='none', num_images = 4):

    from Pre.utils import JsonDatasetNIm as JsonDataset



    print('has cuda?', cuda)
    #images = [f for f in os.listdir(test_folder) if f.endswith('.png')]
    #images.sort(key=lambda name: int(name.split('.png')[0]))
    # loadtest lables function

    PPred_time = time_gap
    NNim = num_images

    XX = 54
    YY = 96

    test_labels = loadTestLabels(test_folder, model_type, PPred_time, 1, 50, 0.0 , 0.0 , 1.0)
    n_test = len(test_labels)
    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    test_loader = th.utils.data.DataLoader(
                                            JsonDataset(
                                                        test_labels,
                                                        preprocess=False,
                                                        folder_prefix=test_folder,
                                                        random_trans=0,
                                                        sequence=True,
                                                        pred_time = PPred_time,
                                                        frame_interval = 12,
                                                        N_im = NNim),
                                           batch_size=32,
                                           shuffle= True,
                                           **kwargs)

    numChannel = test_loader.dataset.numChannel
    if model_type == "cnn":
       model = ConvolutionalNetwork(num_channel=numChannel)
    elif model_type == "CNN_LSTM":
       model = CNN_LSTM(num_channel=numChannel)
    else:
       raise ValueError("Model type not supported")

    model.load_state_dict(th.load(weights))
    # IMPORTANT, if drop rate isn't 0, without this prediction is bad
    model.eval()
    # Move variables to gpu
    if cuda:
       model.cuda()

    predictions = {}
    idx = 0
    test_loss = 0.0
    # Loss functions
    loss_fn = nn.MSELoss(size_average=False)
    test_size = n_test
    start_time = time.time()
    print(test_size, 'images')
    print('start testing ...')
    for i, (inputs, targets) in enumerate(test_loader):
        # Move variables to gpu
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        test_loss += loss.item()
        #row = prediction.shape[0]
        #for i in range(row):
        #    if cuda:
        #        ls = prediction[i].cpu().detach().numpy().tolist()
        #    else: ls = prediction[i].detach().numpy().tolist()
        #    predictions.update({str(int(test_labels[idx])+time_gap):ls})
        #    idx += 1
    print('test error is:', test_loss / test_size)
    #jsObj = json.dumps(predictions)
    with open("./Pre/testError.txt", "a") as f:
        f.write("current model: ")
        f.write(model_type)
        f.write("\n")
        f.write("time gap is:")
        f.write(str(time_gap))
        f.write("\ntest error:")
        f.write(str(test_loss / test_size))
        f.write("\n\n")
    f.close()
    #out = ''
    #if output == 'none':
    #    out = './Pre/results/predictions_'+model_type+"_"+str(time_gap)+'.json'
    #else:
    #    out = output
    #fileObject = open(out, 'w')
    #fileObject.write(jsObj)
    #fileObject.close()
    print("Total train time: {:.2f} s".format((time.time() - start_time)))
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('-f', '--test_folder', help='Test folder', type=str, default='Pre/3dmodel/test_1_episode_')
    parser.add_argument('-bs', '--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-w', '--weight', help='Model weight', default='Pre/results/cnn_model_15s_6im_tmp.pth')
    parser.add_argument('-m', '--model', help='Model type', type=str, default='cnn', choices=['cnn','CNN_RNN','CNN_LSTM'])
    parser.add_argument('-o', '--output', help='output file', default='none')
    parser.add_argument('-t', '--time_gap', help='time gap', default=15, type=int)
    parser.add_argument('-cuda', '--cuda', action='store_true', help='enable cuda', default=True)
    parser.add_argument('-ni', '--num_images', help='Number of images in the series', default=6, type=int)
    args = parser.parse_args()

    main(test_folder=args.test_folder, batchsize=args.batch_size, weights=args.weight,
         cuda=args.cuda, model_type=args.model, output=args.output, time_gap=args.time_gap, num_images = args.num_images )
