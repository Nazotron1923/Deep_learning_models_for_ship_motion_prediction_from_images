"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import json
import os
from datetime import datetime
import random
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.utils import loadLabels, gen_dict_for_json, write_result
from Pre.models import ConvolutionalNetwork_p1, ConvolutionalNetwork_p2, CNN_stack_PR_FC
from Pre.utils import JsonDatasetNIm_CNN_2 as JsonDataset

"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc


def main(train_folder, val_folder=None, num_epochs=100, batchsize=600,
         learning_rate=0.0001, seed=42, cuda=True, num_output=2, random_trans=0.5,
         model_type="CNN_stack_PR_FC", evaluate_print=1, load_model="", time_gap=5, num_images = 4, frame_interval = 12):

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
    base_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_PR_FC_predict_"+str(time_gap)+ "s_"+str(today)
    ress_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_PR_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/result"
    lable_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_PR_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/labels"
    weight_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_PR_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/weight"
    img_dir = "./Pre/results/train_CNN_"+str(num_images)+"_images_stack_PR_FC_predict_"+str(time_gap)+ "s_"+str(today)+ "/img"
    train_val_log_file = "train_val_log_file"
    os.mkdir(base_dir)
    os.mkdir(ress_dir)
    os.mkdir(lable_dir)
    os.mkdir(weight_dir)
    os.mkdir(img_dir)

    seq_per_ep = int(360/num_images)


    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadLabels(train_folder, 1, 32, seq_per_ep)
    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_labels), len(val_labels), len(test_labels)

    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
                                            JsonDataset(train_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        pred_time = time_gap,
                                                        frame_interval = 12,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)

    # Random transform also for val ?
    val_loader = th.utils.data.DataLoader(
                                            JsonDataset(val_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        pred_time = time_gap,
                                                        frame_interval = 12,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )

    test_loader = th.utils.data.DataLoader(
                                            JsonDataset(test_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        random_trans=0,
                                                        pred_time = time_gap,
                                                        frame_interval = 12,
                                                        N_im = num_images),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs)
    num_im_to_predict = int(24/frame_interval)*time_gap
    n_train, n_val, n_test = len(train_loader), len(val_loader), len(test_loader)

    model = CNN_stack_PR_FC(num_channel=3*num_images, cnn_fc_size = 1024 + num_images*2, num_output=num_im_to_predict*2 )

    if cuda:
        model.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers

    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)

    # Loss functions
    loss_fn = nn.MSELoss(reduction = 'sum')
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_val_error = np.inf
    best_train_error = np.inf
    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []

    model_name = "/CNN_{}_images_stack_PR_FC_predict_{}_s_tmp.pth".format(num_images, time_gap)
    model_name_dict = {}
    best_model_path = weight_dir + model_name

    tmp_str = '/CNN_'+str(num_images)+'_images_stack_PR_FC_predict_'+str(time_gap)+'_s_lr_'+str(learning_rate)


    fig = plt.figure(figsize=(22,15))
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='val loss')
    plt.legend(loc='upper right', fontsize=18)
    fig.canvas.draw()
    plt.title("train epochs - loss")
    plt.xlabel("epochs", fontsize = 18)
    plt.ylabel("loss", fontsize = 18)
    plt.show(block=False)
    # Finally, launch the training loop.
    start_time = time.time()


    print("Starting training...")
    # We iterate over epochs:

    for epoch in tqdm(range(num_epochs)):
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for k, (inputs, p_and_roll, targets) in enumerate(train_loader):

            if cuda:
                inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()

            # Convert to pytorch variables
            inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

            optimizer.zero_grad()

            predictions = model(inputs, p_and_roll, num_images, cuda)
            loss = loss_fn(predictions, targets)/ num_im_to_predict

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_error = (train_loss / (n_train*batchsize))*100

        # Do a full pass on validation data
        with th.no_grad():
            model.eval()
            for inputs, p_and_roll, targets in val_loader:

                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                predictions = model(inputs, p_and_roll, num_images, cuda)
                loss = loss_fn(predictions, targets)/ num_im_to_predict
                val_loss += loss.item()

            # Compute error per sample
            val_error = (val_loss / (n_val*batchsize))*100
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
            fig.show()
            json.dump(train_err_list, open(ress_dir+tmp_str+"train_loss.json",'w'))
            json.dump(val_err_list, open(ress_dir+tmp_str+"val_loss.json",'w'))
            print("  training loss:\t\t{:.6f}".format(train_error))
            print("  validation loss:\t\t{:.6f}".format(val_error))


    plt.savefig(img_dir+'/CNN_'+str(num_images)+'_images_stack_PR_FC_predict_'+str(time_gap) +'_s_loss.png')



    model.load_state_dict(th.load(best_model_path))

    test_loss = 0.0

    with th.no_grad():
        key = 0
        origins = [{} for i in range(num_im_to_predict)]
        origin_names = [lable_dir+'/CNN_'+str(num_images)+'_images_stack_PR_FC_origin_predict_'+str(i+1)+':'+str(num_im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(num_im_to_predict)]
        preds = [{} for i in range(num_im_to_predict)]
        pred_names = [lable_dir+'/CNN_'+str(num_images)+'_images_stack_PR_FC_pred_predict_'+str(i+1)+':'+str(num_im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(num_im_to_predict)]

        for inputs, p_and_roll, targets  in test_loader:
            if cuda:
                inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()

            inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)



            predictions = model(inputs, p_and_roll, num_images, cuda)

            key_tmp = np.linspace(key*batchsize , (key+1)*batchsize, batchsize, dtype =int )
            for pred_im in range(num_im_to_predict):
                tmp1 = gen_dict_for_json(key_tmp, targets[:,pred_im,:].cpu())
                tmp2 = gen_dict_for_json(key_tmp, predictions[:,pred_im,:].cpu())

                origins[pred_im] = {**origins[pred_im], **tmp1}
                preds[pred_im] = {**preds[pred_im], **tmp2}

            loss = loss_fn(predictions, targets)/ num_im_to_predict
            test_loss += loss.item()
            key+=1

        for i in range(num_im_to_predict):
            json.dump(preds[i], open(pred_names[i],'w'))
            json.dump(origins[i], open(origin_names[i],'w'))

    test_error = (test_loss /(n_test*batchsize))*100

    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(min(val_err_list)))
    print("  test loss:\t\t\t{:.6f}".format(test_error))

    # write result into result.txt

    final_time = (time.time() - start_time)/60
    print("Total train time: {:.2f} mins".format(final_time))
    write_result( ress_dir + "/result.txt", model_type, best_train_error, best_val_error,
                        test_error, time_gap, num_images/2, seq_per_ep, num_images, frame_interval, batchsize, seed, n_train, n_val,
                        n_test, num_epochs, [model], [optimizer], final_time)




    return best_train_error, best_val_error, test_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('-vf', '--val_folder', help='Validation folder', type=str)
    parser.add_argument('--num_epochs', help='Number of epoch', default= 50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default= 64, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn', 'CNN_LSTM'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-t', '--time_gap', help='Time gap', default=1, type=int)
    parser.add_argument('-ni', '--num_images', help='Number of images in the series', default=2, type=int)
    parser.add_argument('-bt', '--big_test', help='Test hyperparameters', default=0, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()

    if (args.big_test == 0):
        main(train_folder=args.train_folder, val_folder=args.val_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
             learning_rate=args.learning_rate, cuda=args.cuda,
             seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=args.time_gap, num_images = args.num_images )
    else:
        today = datetime.now()
        base_dir = "./Pre/results/BT_train_CNN_n_images_stack_PR_FC_"+str(today)
        os.mkdir(base_dir)

        parm0_lr = []
        parm1_time_gap = []
        parm2_num_images = []
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []
        time_gap_p = [20, 15, 10, 5, 2, 1]
        lr_p = [1e-6, 5e-6, 1e-5, 5e-5]
        nn_images = [10, 8, 6, 4]
        for lr in lr_p:
            for ii in nn_images:
                for tg in time_gap_p:
                    tmp_train, tmp_val, tmp_test = main(train_folder=args.train_folder, val_folder=args.val_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
                         learning_rate=lr, cuda=args.cuda,
                         seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=tg, num_images = ii )
                    parm0_lr.append(lr)
                    parm1_time_gap.append(tg)
                    parm2_num_images.append(ii)
                    parm3_best_train_loss.append(tmp_train)
                    parm4_best_val_loss.append(tmp_val)
                    parm5_best_test_loss.append(tmp_test)

                plt.figure(lr*1000000 + ii)
                # resize pic to show details
                plt.figure(figsize=(20, 12))
                plt.plot(time_gap_p, parm5_best_test_loss, 'r-', label='test error')
                plt.title("test_error - time_gap")
                plt.xlabel("Time_gap")
                plt.ylabel("Test_error")
                plt.legend(loc='upper right')
                plt.savefig(base_dir+'/error_test_use_'+str(ii)+'im_lr_'+str(lr)+'.png')


                for x in range(len(parm3_best_train_loss)):
                    print("---------------------------------------")
                    print("lr       - > ", parm0_lr[x])
                    print("time_gap - > ", parm1_time_gap[x])
                    print("number_im -> ", parm2_num_images[x])
                    print("train     ->", parm3_best_train_loss[x])
                    print("val       ->", parm4_best_val_loss[x])
                    print("test      ->", parm5_best_test_loss[x])

                parm0_lr = []
                parm1_time_gap = []
                parm2_num_images = []
                parm3_best_train_loss = []
                parm4_best_val_loss = []
                parm5_best_test_loss = []
