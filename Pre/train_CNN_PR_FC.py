"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import json
import random
from Pre.constants import INPUT_WIDTH, INPUT_HEIGHT, RES_DIR, ONE_IMG_ONLY, DATASET_SEQUENCE
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.utils import loadOriginalLabels_CNN
from Pre.models import ConvolutionalNetwork_p1, ConvolutionalNetwork_p2
from Pre.utils import JsonDatasetNIm_CNN as JsonDataset

"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
from Pre.data_aug import imgTransform
import scipy.misc

evaluate_print = 1  # Print info every 1 epoch
# VAL_BATCH_SIZE = 600  # Batch size for validation and test data
# lam = 0
# total variance lossing
def reg_loss(tensorArray):
    row, col = tensorArray.shape
    total_loss = 0.0
    for i in range(row-1):
        total_loss = total_loss + abs(tensorArray[i+1][0]-tensorArray[i][0])+abs(tensorArray[i+1][1]-tensorArray[i][1])
    return total_loss

def gen_dict_for_json(keys, values):
    d = {}
    for i in range(len(values)):
        d [str(keys[i])] = list(np.array(values[i].numpy(), dtype=float))
    return d


def main(train_folder, val_folder=None, num_epochs=100, batchsize=600,
         learning_rate=0.0001, seed=42, cuda=True, num_output=2, random_trans=0.5,
         model_type="cnn", evaluate_print=1, load_model="", time_gap=5, num_images = 4, frame_interval = 12):

    # Seed the random generator

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

    print('has cuda?', cuda)

    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadOriginalLabels_CNN(train_folder, 1, 320, num_images)

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


    CNN_p = ConvolutionalNetwork_p1(num_channel=3, num_output=1024)
    # 1024 + pitch + roll = 1026
    FC_p = ConvolutionalNetwork_p2(input_size = num_images*1026, num_output=num_im_to_predict*2)

    if cuda:
        CNN_p.cuda()
        FC_p.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_CNN_p = th.optim.SGD(CNN_p.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)
    optimizer_FC_p = th.optim.SGD(FC_p.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=weight_decay, nesterov=True)

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
    CNN_p_name = "CNN_FC_model_cnn_p_{}s_using_{}im_tmp.pth".format(time_gap, num_images)
    best_CNN_p_path = RES_DIR + CNN_p_name
    FC_p_name = "CNN_FC_model_fc_p_{}s_using_{}im_tmp.pth".format(time_gap, num_images)
    best_FC_p_path = RES_DIR + FC_p_name
    # setup figure parameters
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
        CNN_p.train()
        FC_p.train()
        train_loss, val_loss = 0.0, 0.0

        for k, (inputs, p_and_roll, targets) in enumerate(train_loader):

            if cuda:
                inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()

            # Convert to pytorch variables
            inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

            optimizer_CNN_p.zero_grad()
            optimizer_FC_p.zero_grad()


            features = [CNN_p(inputs[:,i,:,:,:]) for i in range(num_images-1, -1, -1)]
            PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]

            input_fc = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(num_images)]
            input_fc = th.cat(input_fc, 2).view(inputs.size(0), 1, -1)

            predictions = FC_p(input_fc).view(inputs.size(0), -1, 2)
            # print("prediction -- ", predictions.size() )
            # print("targets    -- ", targets.size() )
            loss = loss_fn(predictions, targets)/ num_im_to_predict

            loss.backward()
            train_loss += loss.item()
            optimizer_CNN_p.step()
            optimizer_FC_p.step()
        # Do a full pass on validation data
        with th.no_grad():
            CNN_p.eval()
            FC_p.eval()
            for inputs, p_and_roll, targets in val_loader:

                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                features = [CNN_p(inputs[:,i,:,:,:]) for i in range(num_images-1, -1, -1)]
                PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]

                input_fc = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(num_images)]
                input_fc = th.cat(input_fc, 2).view(inputs.size(0), 1, -1)

                predictions = FC_p(input_fc).view(inputs.size(0), -1, 2)
                loss = loss_fn(predictions, targets)/ num_im_to_predict
                val_loss += loss.item()

            # Compute error per sample
            val_error = val_loss / (n_val*batchsize)
            if val_error < best_error:
                best_error = val_error
                # Move back weights to cpu
                if cuda:
                    CNN_p.cpu()
                    FC_p.cpu()
                # Save Weights
                th.save(CNN_p.state_dict(), best_CNN_p_path)
                th.save(FC_p.state_dict(), best_FC_p_path)

                if cuda:
                    CNN_p.cuda()
                    FC_p.cuda()
            if (train_loss / (n_train*batchsize)) < best_train_error:
                best_train_error = train_loss / (n_train*batchsize)

        if (epoch + 1) % evaluate_print == 0:
            # update figure value and drawing
            train_l = train_loss / (n_train*batchsize)
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
            # time.sleep(0.01)
            fig.show()

            print("  training loss:\t\t{:.6f}".format(train_loss / (n_train*batchsize)))
            print("  validation loss:\t\t{:.6f}".format(val_error))


    print('---------------------------------------TEST---------------------------')

    plt.savefig(RES_DIR+'CNN_FC_1_predict_'+str(time_gap) +'s_using_'+str(num_images)+'im_loss.png')
    # After training, we compute and print the test error:
    CNN_p.load_state_dict(th.load(best_CNN_p_path))
    FC_p.load_state_dict(th.load(best_FC_p_path))

    test_loss = 0.0

    with th.no_grad():
        key = 0
        origins = [{} for i in range(num_im_to_predict)]
        origin_names = [RES_DIR+'CNN_FC_1_origin_label_use_'+str(num_images)+'_to_predict_'+str(i+1)+':'+str(num_im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(num_im_to_predict)]
        preds = [{} for i in range(num_im_to_predict)]
        pred_names = [RES_DIR+'CNN_FC_1_pred_label_use_'+str(num_images)+'_to_predict_'+str(i+1)+':'+str(num_im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(num_im_to_predict)]

        for inputs, p_and_roll, targets  in test_loader:
            if cuda:
                inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()

            inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)


            features = [CNN_p(inputs[:,i,:,:,:]) for i in range(num_images-1, -1, -1)]
            PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]

            input_fc = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(num_images)]
            input_fc = th.cat(input_fc, 2).view(inputs.size(0), 1, -1)

            predictions = FC_p(input_fc).view(inputs.size(0), -1, 2)

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
    print("Final results:")
    # print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  best validation loss:\t\t{:.6f}".format(min(val_err_list)))
    print("  test loss:\t\t\t{:.6f}".format(test_loss /(n_test*batchsize)))

    # write result into result.txt
    # format fixed because we use this file later in pltModelTimegap.py
    with open("./Pre/result_CNN_FC_1.txt", "a") as f:
        f.write("use images: ")
        f.write(str(num_images))
        f.write("\n")
        f.write("to predict (seconds):")
        f.write(str(time_gap))
        f.write("\nbest train error:")
        f.write(str(best_train_error))
        f.write("\nbest validation loss:")
        f.write(str(best_error))
        f.write("\nfinal test loss:")
        f.write(str(test_loss / (n_test*batchsize)))

        f.write("\n")
        f.write("Number of images in the series:")
        f.write(str(num_images))
        f.write("\n")
        f.write("Train examples:")
        f.write(str(n_train*batchsize*num_images))
        f.write("\n")
        f.write("Val examples:")
        f.write(str(n_val*batchsize*num_images))
        f.write("\n")
        f.write("Test examples:")
        f.write(str(n_test*batchsize*num_images))
        f.write("\n")
        f.write("\n\n")
    f.close()
    print("Total train time: {:.2f} mins".format((time.time() - start_time)/60))
    return best_train_error, best_error, test_loss / (n_test*batchsize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('-vf', '--val_folder', help='Validation folder', type=str)
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=4, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn', 'CNN_LSTM'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-t', '--time_gap', help='Time gap', default=5, type=int)
    parser.add_argument('-ni', '--num_images', help='Number of images in the series', default=6, type=int)
    parser.add_argument('-bt', '--big_test', help='Test hyperparameters', default=0, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()

    if (args.big_test == 0):
        main(train_folder=args.train_folder, val_folder=args.val_folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
             learning_rate=args.learning_rate, cuda=args.cuda,
             seed=args.seed, load_model=args.load_model, model_type=args.model_type, time_gap=args.time_gap, num_images = args.num_images )
    else:
        parm0_lr = []
        parm1_time_gap = []
        parm2_num_images = []
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []
        time_gap_p = [1, 2, 5, 10, 15, 20]
        lr_p = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
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
                plt.title("t_error - time_gap")
                plt.xlabel("Time_gap")
                plt.ylabel("Test_error")
                plt.legend(loc='upper right')
                plt.savefig(RES_DIR+'_KAMINSKYI_error_test_use_'+str(ii)+'im_lr_'+str(lr)+'.png')


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
