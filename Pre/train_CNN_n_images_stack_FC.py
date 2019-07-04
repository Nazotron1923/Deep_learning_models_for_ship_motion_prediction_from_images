"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
from datetime import datetime
import os
import random
import json
from Pre.constants import MAX_WIDTH, MAX_HEIGHT, RANDS
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from Pre.utils import loadTrainLabels, gen_dict_for_json, write_result
from Pre.utils import JsonDatasetNIm as JsonDataset
# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.models import ConvolutionalNetwork
"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc


def main(train_folder, num_epochs=50, batchsize=600, learning_rate=0.0001, seed=42,
            cuda=True, num_output=2, random_trans=0.0, model_type="CNN_n_images_stack_FC",
            evaluate_print=1, load_model="", time_gap=5, num_images = 4, frame_interval = 12):

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
    base_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_FC_predict_"+str(time_gap)+ "s_"+str(today)
    res_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/result"
    lable_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/labels"
    img_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/img"
    os.mkdir(base_dir)
    os.mkdir(res_dir)
    os.mkdir(lable_dir)
    os.mkdir(img_dir)

    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadTrainLabels(train_folder, model_type, time_gap, 1, 320)

    # Retrieve number of samples per set
    # n_train, n_val, n_test = len(train_labels), len(val_labels), len(test_labels)

    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
                                            JsonDataset(train_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        pred_time = time_gap,
                                                        frame_interval = frame_interval,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)

    val_loader = th.utils.data.DataLoader(
                                            JsonDataset(val_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        pred_time = time_gap,
                                                        frame_interval = frame_interval,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )

    test_loader = th.utils.data.DataLoader(
                                            JsonDataset(test_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        pred_time = time_gap,
                                                        frame_interval = frame_interval,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)

    num_channel = train_loader.dataset.numChannel

    if model_type == "CNN_n_images_stack_FC":
        model = ConvolutionalNetwork(num_channel=num_channel, num_output=num_output)
    else:
        raise ValueError("Model type not supported")

    if cuda:
        model.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)

    # Loss functions
    loss_fn = nn.MSELoss(reduction = 'sum')

    best_val_error = np.inf
    best_train_error = np.inf
    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []
    model_config = "{}_model_predict_{}_s_using_{}_im_lr_{}_tmp.pth".format(model_type, time_gap, num_images, learning_rate)
    best_model_path = res_dir + model_config

    # setup figure parameters
    fig = plt.figure(figsize=(22,15))
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='validation loss')
    plt.legend(loc='upper right', fontsize=18)
    fig.canvas.draw()
    plt.title("Evolution of loss functions")
    plt.xlabel("Epochs", fontsize = 18)
    plt.ylabel("Loss", fontsize = 18)
    plt.show(block=False)
    # Finally, launch the training loop.
    start_time = time.time()


    print("Starting training...")
    # We iterate over epochs:
    null_tempate_1 = th.zeros(num_channel, MAX_HEIGHT,MAX_WIDTH, dtype = th.float32 )
    null_tempate_2 =  th.zeros(2, dtype = th.float32 )


    for epoch in tqdm(range(num_epochs)):
        n_train = 0
        n_val = 0

        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0
        # Full pass on training data
        # Update the model after each minibatch
        for i, (inputs, targets) in enumerate(train_loader):

            # if we have incomplete mini_time_series we will skip it

            if( th.all(th.eq(inputs[0], null_tempate_1))  and  th.all(th.eq(targets[0], null_tempate_2))):
                continue
            n_train += targets.size(0)

            # Move variables to
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_error = train_loss / n_train
        # Do a full pass on validation data
        with th.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                # if we have incomplete mini_time_series we will skip it
                if( th.all(th.eq(inputs[0], null_tempate_1))  and  th.all(th.eq(targets[0], null_tempate_2))):
                    continue

                n_val += targets.size(0)
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                # Set volatile to True because we don't need to compute gradient
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)# + loss_tmp
                val_loss += loss.item()

            # Compute error per sample
            val_error = val_loss / n_val
            if val_error < best_val_error:
                best_val_error = val_error
                # Move back weights to cpu
                if cuda:
                    model.cpu()
                # Save Weights
                th.save(model.state_dict(), best_model_path)
                if cuda:
                    model.cuda()
            if train_error < best_train_error:
                best_train_error = train_error

        if (epoch + 1) % evaluate_print == 0:
            # update figure value and drawing
            xdata.append(epoch+1)
            train_err_list.append(train_error)
            val_err_list.append(val_error)
            li.set_xdata(xdata)
            li.set_ydata(train_err_list)
            l2.set_xdata(xdata)
            l2.set_ydata(val_err_list)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            # time.sleep(0.01)
            fig.show()
            # Then we print the results for this epoch:
            # Losses are averaged over the samples
            # print("Epoch {} of {} took {:.3f}s".format(
            #     epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_error))
            print("  validation loss:\t\t{:.6f}".format(val_error))

    print('---------------------------------------TEST---------------------------')
    plt.savefig(img_dir+'/CNN_'+str(num_images)+'_images_stack_FC'+'_predict_'+str(time_gap) +'s_lr'+str(learning_rate)+'_loss.png')
    # After training, we compute and print the test error:
    model.load_state_dict(th.load(best_model_path))
    test_loss = 0.0
    n_test = 0

    with th.no_grad():

        origin = {}
        origin_name = lable_dir+'/CNN_'+str(num_images)+'_images_stack_FC_origin_label_predict_'+str(time_gap)+'s_lr_'+str(learning_rate)+'.json'
        pred = {}
        pred_name = lable_dir+'/CNN_'+str(num_images)+'_images_stack_FC_pred_label_predict_'+str(time_gap)+'s_lr_'+str(learning_rate)+'.json'

        for i , (inputs, targets) in enumerate(test_loader):
            if( th.all(th.eq(inputs[0], null_tempate_1))  and  th.all(th.eq(targets[0], null_tempate_2))):
                continue
            n_test += targets.size(0)
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = model(inputs)

            key_tmp = np.linspace(i*batchsize, (i+1)*batchsize - 1, batchsize, dtype =int )
            tmp1 = gen_dict_for_json(key_tmp, targets.cpu())
            tmp2 =  gen_dict_for_json(key_tmp, predictions.cpu())

            origin = {**origin, **tmp1}
            pred = {**pred, **tmp2}
            loss = loss_fn(predictions, targets)# + loss_tmp
            test_loss += loss.item()


        json.dump(pred, open(pred_name,'w'))
        json.dump(origin, open(origin_name,'w'))
    test_error = test_loss / n_test
    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(min(val_err_list)))
    print("  test loss:\t\t\t{:.6f}".format(test_error))

    final_time = (time.time() - start_time)/60
    print("Total train time: {:.2f} mins".format(final_time))

    # write result into ./Pre/result.txt
    write_result( res_dir + "/result.txt", model_type, best_train_error, best_val_error,
                        test_error, time_gap, True, num_images, 'all epizode', 1, frame_interval, batchsize, seed, n_train, n_val,
                        n_test, num_epochs, [model], [optimizer], final_time)


    return best_train_error, best_val_error, test_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=600, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('--model_type', help='Model type: cnn', default="CNN_n_images_stack_FC", type=str, choices=['cnn', 'CNN_LSTM'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-t', '--time_gap', help='Time gap', default=10, type=int)
    parser.add_argument('-ni', '--num_images', help='Number of images in the series', default=6, type=int)
    parser.add_argument('-bt', '--big_test', help='Test hyperparameters', default=0, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()

    if (args.big_test == 0):
        main(train_folder=args.train_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
             learning_rate=args.learning_rate, cuda=args.cuda,
             seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=args.time_gap, num_images = args.num_images )
    else:
        parm1_time_gap = []
        parm2_num_images = []
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []
        time_gap_p = [1, 2, 5, 10, 15, 20, 25, 30]
        nn_images = [6, 4]

        for ii in nn_images:
            for tg in time_gap_p:
                tmp_train, tmp_val, tmp_test = main(train_folder=args.train_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
                     learning_rate=args.learning_rate, cuda=args.cuda,
                     seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=tg, num_images = ii )
                parm1_time_gap.append(tg)
                parm2_num_images.append(ii)
                parm3_best_train_loss.append(tmp_train)
                parm4_best_val_loss.append(tmp_val)
                parm5_best_test_loss.append(tmp_test)

            plt.figure(ii)
            # resize pic to show details
            plt.figure(figsize=(20, 12))
            plt.plot(time_gap_p, parm5_best_test_loss, 'r-', label='test error')
            plt.title("t_error - time_gap")
            plt.xlabel("Time_gap")
            plt.ylabel("Test_error")
            plt.legend(loc='upper right')
            plt.savefig('-----KAMINSKYI_error_test_'+str(ii)+'im_pitch_final'+'.png')


            for x in range(len(parm3_best_train_loss)):
                print("---------------------------------------")
                print("time_gap - > ", parm1_time_gap[x])
                print("number_im -> ", parm2_num_images[x])
                print("train     ->", parm3_best_train_loss[x])
                print("val       ->", parm4_best_val_loss[x])
                print("test      ->", parm5_best_test_loss[x])

            parm1_time_gap = []
            parm2_num_images = []
            parm3_best_train_loss = []
            parm4_best_val_loss = []
            parm5_best_test_loss = []
