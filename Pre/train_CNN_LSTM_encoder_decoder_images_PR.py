"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
from datetime import datetime
import json
import os
import random
import numpy as np
import torch as th

import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torchvision import transforms


from Pre.constants import RES_DIR, LEN_SEQ, SEQ_PER_EPISODE_C
from Pre.utils import loadLabels, gen_dict_for_json, write_result
# from Pre.utils import JsonDatasetForLSTM as JsonDataset
from Pre.utils import JsonDataset_universal as JsonDataset
from Pre.models import LSTM_encoder , LSTM_decoder, AutoEncoder, CNN_LSTM_encoder_decoder_images_PR

# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def train(cuda, inputs, targets, model, optimizer, criterion,
        predict_n_pr, use_n_im):

    encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                    model.initHiddenEncoder(inputs.size(0)).cuda())

    decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                    model.initHiddenDecoder(targets.size(0)).cuda())

    optimizer.zero_grad()

    target_length = LEN_SEQ - predict_n_pr - use_n_im
    loss = 0

    for im in range(use_n_im-1, target_length+use_n_im):

        image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
        pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

        # print("image_s[0].size() -- ", image_s[0].size())
        # print("pr_s[0].size() -- ", pr_s[0].size())
        prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)

        # print("prediction.size() -- ", prediction.size())
        # print("targets[:,im+1:im+predict_n_pr+1,:].size() -- ", targets[:,im+1:im+10+1,:].size())
        # print("targets[:,,:].size() -- ", targets.size())
        # print("predict_n_pr--", predict_n_pr)
        loss += criterion(prediction, targets[:,im+1 : im+predict_n_pr+1,:])/predict_n_pr


    loss.backward()
    optimizer.step()

    return loss.item() / target_length


def eval(cuda, inputs, targets, model, criterion, predict_n_pr, use_n_im):

    encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                    model.initHiddenEncoder(inputs.size(0)).cuda())

    decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                    model.initHiddenDecoder(targets.size(0)).cuda())

    target_length = LEN_SEQ - predict_n_pr - use_n_im

    with th.no_grad():
        loss = 0

        for im in range(use_n_im-1, target_length+use_n_im):

            image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
            pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

            prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)

            loss += criterion(prediction, targets[:,im+1:im+predict_n_pr+1,:])/predict_n_pr

    return loss.item() / target_length


def test(cuda, i, origins, preds, batchsize, inputs, targets,
        model,
        criterion, predict_n_pr, use_n_im):

    encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                    model.initHiddenEncoder(inputs.size(0)).cuda())

    decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                    model.initHiddenDecoder(targets.size(0)).cuda())

    target_length = LEN_SEQ - predict_n_pr - use_n_im

    with th.no_grad():
        loss = 0

        for im in range(use_n_im-1, target_length+use_n_im):

            image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
            pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

            prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)

            loss += criterion(prediction, targets[:,im+1:im+predict_n_pr+1,:])/predict_n_pr

            key_tmp = np.linspace(i*target_length*batchsize + (im-use_n_im+1)*batchsize , i*target_length*batchsize + (im-use_n_im+2)*batchsize - 1, batchsize, dtype =int )

            for pred_im in range(predict_n_pr):
                tmp1 = gen_dict_for_json(key_tmp, targets[:,im+pred_im+1,:].cpu())
                tmp2 = gen_dict_for_json(key_tmp, prediction[:,pred_im,:].cpu())

                origins[pred_im] = {**origins[pred_im], **tmp1}
                preds[pred_im] = {**preds[pred_im], **tmp2}

    return loss.item() / target_length, origins, preds


def main(train_folder, num_epochs=50, batchsize=32, learning_rate=0.0001, seed=42, load_weight = False,
        cuda=True, model_type="CNN_LSTM_encoder_decoder_images_PR", evaluate_print=1,
        time_gap=5, frame_interval = 12, sec_to_pred = 5):

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


    if load_weight:
        base_dir = "./Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_5s_using_5s_2019-07-02 09:26:50.520968"
        ress_dir = base_dir +  "/result"
        lable_dir = base_dir + "/labels"
        img_dir = base_dir +  "/img"
    else:
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_dir = "./Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_"+str(time_gap)+ "s_using_"+str(sec_to_pred)+"s_"+today
        ress_dir = "./Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_"+str(time_gap)+ "s_using_"+str(sec_to_pred)+"s_"+today+ "/result"
        lable_dir = "./Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_"+str(time_gap)+ "s_using_"+str(sec_to_pred)+"s_"+today+ "/labels"
        img_dir = "./Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_"+str(time_gap)+ "s_using_"+str(sec_to_pred)+"s_"+today+ "/img"

        os.mkdir(base_dir)
        os.mkdir(ress_dir)
        os.mkdir(lable_dir)
        os.mkdir(img_dir)


    train_labels, val_labels, test_labels = loadLabels(train_folder, 1, 320, SEQ_PER_EPISODE_C, p_train=0.7, p_val=0.15, p_test=0.15)
    print(train_labels)
    print(val_labels)
    print(test_labels)


    im_in_one_second = int(24/frame_interval)
    predict_n_pr = im_in_one_second*time_gap
    use_n_im = im_in_one_second*sec_to_pred
    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loaders
    # train_loader = th.utils.data.DataLoader(
    #                                         JsonDataset(train_labels,
    #                                                     preprocess=True,
    #                                                     folder_prefix=train_folder),
    #                                         batch_size=batchsize,
    #                                         shuffle=True,
    #                                         **kwargs)
    #
    # # Random transform also for val ?
    # val_loader = th.utils.data.DataLoader(
    #                                         JsonDataset(val_labels,
    #                                                     preprocess=True,
    #                                                     folder_prefix=train_folder),
    #                                         batch_size=batchsize,
    #                                         shuffle=True,
    #                                         **kwargs
    #                                     )
    #
    # test_loader = th.utils.data.DataLoader(
    #                                         JsonDataset(test_labels,
    #                                                     preprocess=True,
    #                                                     folder_prefix=train_folder),
    #                                         batch_size=batchsize,
    #                                         shuffle=True,
    #                                         **kwargs)

    train_loader = th.utils.data.DataLoader(
                                                JsonDataset(train_labels,
                                                            preprocess=True,
                                                            folder_prefix=train_folder,
                                                            predict_n_im = predict_n_pr,
                                                            use_n_im = use_n_im,
                                                            seq_per_ep = SEQ_PER_EPISODE_C,
                                                            use_LSTM = True,
                                                            use_stack = False),
                                                batch_size=batchsize,
                                                shuffle=True,
                                                **kwargs
                                            )

    val_loader = th.utils.data.DataLoader(
                                            JsonDataset(val_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        predict_n_im = predict_n_pr,
                                                        use_n_im = use_n_im,
                                                        seq_per_ep = SEQ_PER_EPISODE_C,
                                                        use_LSTM = True,
                                                        use_stack = False),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )

    test_loader = th.utils.data.DataLoader(
                                            JsonDataset(test_labels,
                                                        preprocess=True,
                                                        folder_prefix=train_folder,
                                                        predict_n_im = predict_n_pr,
                                                        use_n_im = use_n_im,
                                                        seq_per_ep = SEQ_PER_EPISODE_C,
                                                        use_LSTM = True,
                                                        use_stack = False),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )

    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_loader), len(val_loader), len(test_loader)


    model = CNN_LSTM_encoder_decoder_images_PR( encoder_input_size = use_n_im*1026, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 2*predict_n_pr)
    # LSTM_encoder_p = LSTM_encoder(input_size=use_n_im*1026, hidden_size=1024, num_layers=1)
    # LSTM_decoder_total = LSTM_decoder(hidden_size=1024, output_size = 2*predict_n_pr)
    CNN_p = AutoEncoder()
    CNN_p.load_state_dict(torch.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))
    # mode0l_2.layer[0].weight
    # print("CNN_p.state_dict() ", CNN_p.state_dict())
    model.encoder[0].weight = CNN_p.encoder[0].weight
    model.encoder[0].bias = CNN_p.encoder[0].bias
    model.encoder[3].weight = CNN_p.encoder[3].weight
    model.encoder[3].bias = CNN_p.encoder[3].bias
    model.encoder[6].weight = CNN_p.encoder[6].weight
    model.mu.weight = CNN_p.fc1.weight
    model.mu.bias = CNN_p.fc1.bias
    model.std.weight = CNN_p.fc2.weight
    model.std.bias = CNN_p.fc2.bias

    if load_weight:
        LSTM_encoder_p.load_state_dict(torch.load(ress_dir+"/CNN_LSTM_encoder_decoder_images_PR-Encoder_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)))
        LSTM_decoder_total.load_state_dict(torch.load(ress_dir+"/CNN_LSTM_encoder_decoder_images_PR-Decoder_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)))
        CNN_p.load_state_dict(torch.load(ress_dir+"/CNN_LSTM_encoder_decoder_images_PR-CNN_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)))
    # Freeze model weights
    # for param in CNN_p.parameters():
    #     param.requires_grad = False

    if cuda:
        model.cuda()
        # LSTM_encoder_p.cuda()
        # LSTM_decoder_total.cuda()
        # CNN_p.cuda()
    # L2 penalty
    print("model.state_dict() ", model.state_dict())
    weight_decay = 1e-3
    # Optimizers

    # CNN_optimizer = th.optim.Adam(CNN_p.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # LSTM_encoder_optimizer = th.optim.Adam(LSTM_encoder_p.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # LSTM_decoder_optimizer = th.optim.Adam(LSTM_decoder_total.parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizer = th.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    # Loss and optimizer
    criteration = nn.MSELoss(reduction = 'sum')#nn.NLLLoss()

    best_val_error = np.inf
    best_train_error = np.inf
    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []

    file_model = "/CNN_LSTM_encoder_decoder_images_PR_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    # file_CNN_model = "/CNN_LSTM_encoder_decoder_images_PR-CNN_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    # file_encoder_model = "/CNN_LSTM_encoder_decoder_images_PR-Encoder_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    # file_decoder_model = "/CNN_LSTM_encoder_decoder_images_PR-Decoder_part_predict_{}_s_using_{}_s_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    tmp_str = '/CNN_LSTM_encoder_decoder_images_PR_predict_'+str(time_gap) +'s_using_'+str(sec_to_pred)+'s_lr_'+str(learning_rate)
    #
    # CNN_part_dict = {}
    # encoder_part_dict = {}
    # decoder_part_dict = {}

    best_model = ress_dir + file_model
    # best_model_CNN_part = ress_dir + file_CNN_model
    # best_model_encoder_part = ress_dir + file_encoder_model
    # best_model_decoder_part = ress_dir + file_decoder_model

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
        model.train()
        # LSTM_decoder_total.train()
        # CNN_p.train()
        train_loss, val_loss = 0.0, 0.0
        # Full pass on training data
        # Update the model after each minibatch
        for i, (inputs, targets) in enumerate(train_loader):



            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)
            loss = train(cuda, inputs, targets, model, optimizer, criteration, predict_n_pr, use_n_im)
            train_loss += loss

        train_l = (train_loss / (n_train*batchsize))*100
        model.eval()
        # LSTM_decoder_total.eval()
        # CNN_p.eval()

        with th.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, targets = Variable(inputs), Variable(targets)
                loss = eval(cuda, inputs, targets, model, criteration, predict_n_pr, use_n_im)
                val_loss += loss

            val_l = (val_loss/ (n_val*batchsize))*100

        if val_l < best_val_error:
            best_val_error = val_l

            if cuda:
                model.cpu()
                # LSTM_decoder_total.cpu()
                # CNN_p.cpu()
            # Save Weights of all parts
            # SAVE the best model
            th.save(model.state_dict(), best_model)
            # th.save(LSTM_encoder_p.state_dict(), best_model_encoder_part)
            # th.save(LSTM_decoder_total.state_dict(), best_model_decoder_part)
            # CNN_part_dict = CNN_p.state_dict()
            # encoder_part_dict = LSTM_encoder_p.state_dict()
            # decoder_part_dict = LSTM_decoder_total.state_dict()

            if cuda:
                model.cuda()
                # LSTM_decoder_total.cuda()
                # CNN_p.cuda()

        if train_l < best_train_error:
            best_train_error = train_l

        if (epoch + 1) % evaluate_print == 0:

            xdata.append(epoch+1)
            train_err_list.append(train_l)
            val_err_list.append(val_l)
            li.set_xdata(xdata)
            li.set_ydata(train_err_list)
            l2.set_xdata(xdata)
            l2.set_ydata(val_err_list)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            time.sleep(0.01)
            fig.show()
            json.dump(train_err_list, open(ress_dir+tmp_str+"train_loss.json",'w'))
            json.dump(val_err_list, open(ress_dir+tmp_str+"val_loss.json",'w'))
            print("  training loss:\t\t{:.6f}".format(train_l))
            print("  validation loss:\t\t{:.6f}".format(val_l))


    plt.savefig(img_dir+tmp_str+'_loss.png')




    # LOAD the best model
    model.load_state_dict(th.load(best_model))
    # LSTM_encoder_p.load_state_dict(th.load(best_model_encoder_part))
    # LSTM_decoder_total.load_state_dict(th.load(best_model_decoder_part))

    # After training, we compute and print the test error:
    print('Test starting...')
    test_loss = 0.0

    origins = [{} for i in range(predict_n_pr)]
    origin_names = [lable_dir+'/CNN_LSTM_encoder_decoder_images_PR_origin_label_use_'+str(sec_to_pred)+'_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'_.json' for i in range(predict_n_pr)]
    preds = [{} for i in range(predict_n_pr)]
    pred_names = [lable_dir+'/CNN_LSTM_encoder_decoder_images_PR_pred_label_use_'+str(sec_to_pred)+'_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'_.json' for i in range(predict_n_pr)]

    with th.no_grad():
        for i , (inputs, targets) in enumerate(test_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)


            loss, origins, preds  = test(cuda, i, origins, preds, batchsize, inputs,
                                        targets, model, criteration, predict_n_pr, use_n_im)
            test_loss += loss

    test_l = (test_loss / (n_test*batchsize))*100
    for i in range(predict_n_pr):
        json.dump(preds[i], open(pred_names[i],'w'))
        json.dump(origins[i], open(origin_names[i],'w'))

    print("Final results:")
    print("  best train loss:\t\t{:.6f}".format(best_train_error))
    print("  best validation loss:\t\t{:.6f}".format(best_val_error))
    print("  test loss:\t\t\t{:.6f}".format(test_l))


    final_time = (time.time() - start_time)/60
    print("Total train time: {:.2f} mins".format(final_time))

    # write result into ./Pre/result.txt
    write_result( ress_dir + "/result.txt", model_type, best_train_error, best_val_error,
                        test_l, time_gap, sec_to_pred, SEQ_PER_EPISODE_C, LEN_SEQ, frame_interval, batchsize, seed, n_train*batchsize*LEN_SEQ, n_val*batchsize*LEN_SEQ,
                        n_test*batchsize*LEN_SEQ, num_epochs, [model], [optimizer ], final_time)



    return best_train_error, best_val_error, test_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=16, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-4, type=float)
    parser.add_argument('-t', '--time_gap', help='Time in seconds to predict ', default=5, type=int)
    parser.add_argument('-u', '--sec_to_pred', help='How many seconds take to predict smth ', default=5, type=int)
    parser.add_argument('-bt', '--big_test', help='Test hyperparameters', default=0, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()

    if (args.big_test == 0):
        main(train_folder=args.train_folder,
            num_epochs=args.num_epochs,
            batchsize=args.batchsize,
            learning_rate=args.learning_rate,
            cuda=args.cuda,
            seed=args.seed,
            load_weight = False,
            time_gap=args.time_gap,
            sec_to_pred = args.sec_to_pred)
    else:
        parm1_lr = []
        parm2_sec_to_pred = []
        parm22_pred_s =[]
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []

        sec_to_pred_p = [5]
        pred_p = [5]
        lr_p = [5e-5, 1e-4, 5e-4]
        today = datetime.now()
        base_dir = "./Pre/results/BT_train_CNN_stack_PR_FC_"+str(today)
        os.mkdir(base_dir)

        for pred_t in pred_p:
            for stp in sec_to_pred_p:
                for lr in lr_p:
                    tmp_train, tmp_val, tmp_test = main(
                                                        train_folder=args.train_folder,
                                                        num_epochs=args.num_epochs,
                                                        batchsize=args.batchsize,
                                                        learning_rate=lr,
                                                        cuda=args.cuda,
                                                        seed=args.seed,
                                                        load_weight = False,
                                                        time_gap=pred_t,
                                                        sec_to_pred = stp

                                                        )


                    parm1_lr.append(lr)
                    parm2_sec_to_pred.append(stp)
                    parm22_pred_s.append(pred_t)
                    parm3_best_train_loss.append(tmp_train)
                    parm4_best_val_loss.append(tmp_val)
                    parm5_best_test_loss.append(tmp_test)

                plt.figure(stp+pred_t)
                # resize pic to show details
                plt.figure(figsize=(20, 12))
                plt.plot(lr_p, parm5_best_test_loss, 'r-', label='test error')
                plt.title("t_error - time_gap")
                plt.xlabel("lr_p")
                plt.ylabel("Test_error")
                plt.legend(loc='upper right')
                plt.savefig(RES_DIR+'w_CNN_LSTM_simple_error_test_use_'+str(stp)+'s_to_pred' + str(pred_t) + '_s_PR_final'+'.png')
                plt.savefig(base_dir+'/error_test_use_'+str(stp)+'s_to_pred' + str(pred_t) + '_lr.png')

                for x in range(len(parm3_best_train_loss)):
                    print("---------------------------------------")
                    print("lr - > ", parm1_lr[x])
                    print("sec_to_pred -> ", parm2_sec_to_pred[x])
                    print("pred_time -> ", parm22_pred_s[x])
                    print("train     ->", parm3_best_train_loss[x])
                    print("val       ->", parm4_best_val_loss[x])
                    print("test      ->", parm5_best_test_loss[x])

                parm1_lr = []
                parm2_sec_to_pred = []
                parm22_pred_s =[]
                parm3_best_train_loss = []
                parm4_best_val_loss = []
                parm5_best_test_loss = []
