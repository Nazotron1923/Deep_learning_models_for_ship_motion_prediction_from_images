"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import json
import random
import os
from datetime import datetime
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RES_DIR
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from Pre.utils import loadTrainLabels
import torch.nn.functional as F


# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.models import AutoEncoder
"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from Pre.utils import JsonDatasetNIm as JsonDataset



def imshow(img):
    img = img /255
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# total variance lossing
def reg_loss(tensorArray):
    row, col = tensorArray.shape
    total_loss = 0.0
    for i in range(row-1):
        total_loss = total_loss + abs(tensorArray[i+1][0]-tensorArray[i][0])+abs(tensorArray[i+1][1]-tensorArray[i][1])
    return total_loss





def main(train_folder, num_epochs=100, batchsize=32,
         learning_rate=0.0001, seed=42, cuda=True, num_output=2, random_trans=0.5,
         model_type="VAE", evaluate_print=1, load_model="", time_gap=1, num_images = 1, stat_data_file = ''):


    # indicqte randomseed , so that we will be able to reproduce the result in the future
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    if cuda:
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.enabled = False
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    #---------------------------------------------------------------------------
    print('has cuda?', cuda)

    today = datetime.now()
    base_dir = "./Pre/results/train_VAE_"+str(num_images)+ "image_to_"+str(num_images)+"_image_"+str(today)
    ress_dir = "./Pre/results/train_VAE_"+str(num_images)+ "image_to_"+str(num_images)+"_image_"+str(today)+ "/result"
    lable_dir = "./Pre/results/train_VAE_"+str(num_images)+ "image_to_"+str(num_images)+"_image_"+str(today)+ "/labels"
    img_dir = "./Pre/results/train_VAE_"+str(num_images)+ "image_to_"+str(num_images)+"_image_"+str(today)+ "/img"

    os.mkdir(base_dir)
    os.mkdir(ress_dir)
    os.mkdir(lable_dir)
    os.mkdir(img_dir)

    print('has cuda?', cuda)

    XX = 54
    YY = 96
    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadTrainLabels(train_folder, model_type, 0, 1, 543)

    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_labels), len(val_labels), len(test_labels)


    stat_data = json.load(open(stat_data_file))
    # print('mean -> ', stat_data['mean'])
    # print('std -> ', stat_data['std'])


    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
                                            JsonDataset(train_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        sequence=False,
                                                        pred_time = 0,
                                                        frame_interval = 12,
                                                        N_im = 1),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)

    # Random transform also for val ?
    val_loader = th.utils.data.DataLoader(
                                            JsonDataset(val_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        sequence=False,
                                                        pred_time = 0,
                                                        frame_interval = 12,
                                                        N_im = 1),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )

    test_loader = th.utils.data.DataLoader(
                                            JsonDataset(test_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        sequence=False,
                                                        pred_time = 0,
                                                        frame_interval = 12,
                                                        N_im = 1),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)


    numChannel = train_loader.dataset.numChannel
    print("numChannel _______", numChannel)
    model = AutoEncoder(num_channel=numChannel)
    if cuda:
        model.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
    #                          momentum=0.9, weight_decay=weight_decay, nesterov=True)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Loss functions
    loss_fn = nn.MSELoss(reduction = 'sum')
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_error = np.inf
    best_train_error = np.inf
    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []
    model_name = "autoencoder_model_{}s_{}im_tmp".format(model_type, time_gap, num_images)
    best_model_path = "/{}.pth".format(model_name)
    best_model_path = ress_dir + best_model_path
    # setup figure parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='val loss')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    plt.title("train epochs - loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show(block=False)
    # Finally, launch the training loop.
    start_time = time.time()

    print("Starting training...")
    # We iterate over epochs:
    test1 = th.zeros(numChannel, XX, YY, dtype = th.float32 )
    test2 =  th.zeros(2, dtype = th.float32 )

    for epoch in tqdm(range(num_epochs)):
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0
        # Full pass on training data
        # Update the model after each minibatch
        for i, (inputs, targets) in enumerate(train_loader):
            # if we have incomplete mini_time_series we will skip it
            if( th.all(th.eq(inputs[0], test1))  and  th.all(th.eq(targets[0], test2))):
                print("----GAP---")
                continue
            # Adjust learning rate
            # adjustLearningRate(optimizer, epoch, num_epochs, lr_init=learning_rate,
            #                         batch=i, n_batch=len(train_loader), method='multistep')
            # Move variables to
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)
            features, recon_images = model(inputs, cuda)
            loss = loss_fn(recon_images, inputs)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()


        # Do a full pass on validation data
        with th.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                # if we have incomplete mini_time_series we will skip it
                if( th.all(th.eq(inputs[0], test1))  and  th.all(th.eq(targets[0], test2))):
                    print("----GAP---")
                    continue
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                features, recon_images = model(inputs, cuda)
                loss = loss_fn(recon_images, inputs)
                val_loss += loss.item()

            # Compute error per sample
            val_error = val_loss / n_val
            if val_error < best_error:
                best_error = val_error
                # Move back weights to cpu
                if cuda:
                    model.cpu()
                # Save Weights
                th.save(model.state_dict(), best_model_path)
                if cuda:
                    model.cuda()
            if (train_loss / n_train) < best_train_error:
                best_train_error = train_loss / n_train

        if (epoch + 1) % evaluate_print == 0:
            # update figure value and drawing
            train_l = train_loss / n_train
            xdata.append(epoch+1)
            train_err_list.append(train_l)
            val_err_list.append(val_error)
            li.set_xdata(xdata)
            li.set_ydata(train_err_list)
            l2.set_xdata(xdata)
            l2.set_ydata(val_err_list)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            time.sleep(0.01)
            fig.show()
            # Then we print the results for this epoch:
            # Losses are averaged over the samples
            # print("Epoch {} of {} took {:.3f}s".format(
            #     epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_loss / n_train))
            print("  validation loss:\t\t{:.6f}".format(val_error))
            #pic = to_image(predictions.cpu().data)
            #save_image(pic, './image_{}.png'.format(epoch))


    plt.savefig(img_dir+'/CNN_VAE_log_losses.png')
    # After training, we compute and print the test error:
    model.load_state_dict(th.load(best_model_path))
    test_loss = 0.0

    with th.no_grad():
        for inputs, targets  in test_loader:
            if( th.all(th.eq(inputs[0], test1))  and  th.all(th.eq(targets[0], test2))):
                print("----GAP---")
                continue

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            features, recon_images = model(inputs, cuda)
            # loss_tmp = lam*reg_loss(predictions)
            loss = loss_fn(recon_images, inputs)# + loss_tmp
            test_loss += loss.item()
    print("Final results:")
    # print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  best validation loss:\t\t{:.6f}".format(min(val_err_list)))
    print("  test loss:\t\t\t{:.6f}".format(test_loss / n_test))
    # write result into result.txt
    # format fixed because we use this file later in pltModelTimegap.py

    print("Total train time: {:.2f} mins".format((time.time() - start_time)/60))
    return best_train_error, best_error, test_loss / n_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('-stf', '--stat_data_file', help='stat_data_file json', type=str, default = 'Pre/stat_dataset_4_250episodes.json')
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=64, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn', 'CNN_LSTM'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-4, type=float)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()


    main(train_folder=args.train_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
         learning_rate=args.learning_rate, cuda=args.cuda,
         seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=1, num_images = 1 , stat_data_file = args.stat_data_file)
