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

from Pre.constants import SEQ_PER_EPISODE_C, LEN_SEQ, RES_DIR
# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
from Pre.utils import loadLabels, gen_dict_for_json, write_result

from Pre.utils import JsonDataset_universal as JsonDataset

from Pre.models import ConvolutionalNetwork_p1, ConvolutionalNetwork_p2, CNN_stack_PR_FC, CNN_LSTM_encoder_decoder_images_PR, AutoEncoder, LSTM_encoder_decoder_PR, CNN_LSTM_encoder_decoder_images, CNN_LSTM_decoder_images_PR, CNN_PR_FC

"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from torchvision import transforms
import scipy.misc


def train(inputs, targets, model, optimizer, criterion, predict_n_pr, use_n_im):

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


def eval(inputs, targets, model, criterion, predict_n_pr, use_n_im):

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


def test(i, origins, preds, batchsize, inputs, targets,
        model, criterion, predict_n_pr, use_n_im):

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



def main(train_folder, num_epochs = 50, batchsize = 32,
         learning_rate=0.0001, seed=42, cuda=True, load_weight = False,
         model_type="CNN_LSTM_encoder_decoder_images_PR", evaluate_print=1, time_gap=5, use_sec = 5, frame_interval = 12):

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
        today = "2019-07-02 09:26:50"
    else:
        today = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_dir = "./Pre/results/train_"+ model_type +"_using_" +str(use_sec)+  "_s_to_predict_"+str(time_gap)+ "_s_lr_" + str(learning_rate) + "_" + today
    ress_dir = base_dir+ "/result"
    lable_dir = base_dir+ "/labels"
    weight_dir = base_dir + "/weight"
    img_dir = base_dir + "/img"

    if not load_weight:
        os.mkdir(base_dir)
        os.mkdir(ress_dir)
        os.mkdir(lable_dir)
        os.mkdir(weight_dir)
        os.mkdir(img_dir)

    # parametres general

    im_in_one_second = int(24/frame_interval)
    predict_n_pr = im_in_one_second*time_gap
    use_n_im = im_in_one_second*use_sec
    use_LSTM = False
    use_stack = False
    use_n_channels = 3
    seq_per_ep = SEQ_PER_EPISODE_C

    # parametr for different models
    if 'LSTM' in model_type:
        use_LSTM = True
    else:
        seq_per_ep = int(360/use_n_im)

    if 'stack' in model_type:
        use_stack = True
        use_n_channels = 3*use_n_im

    # parametr for different models

    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadLabels(train_folder, 0, 54, seq_per_ep, p_train=0.7, p_val=0.15, p_test=0.15)

    # Keywords for pytorch dataloader, augment num_workers could work faster
    kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}
    # Create data loaders

    train_loader = th.utils.data.DataLoader(
                                                JsonDataset(train_labels,
                                                            preprocess=True,
                                                            folder_prefix=train_folder,
                                                            predict_n_im = predict_n_pr,
                                                            use_n_im = use_n_im,
                                                            seq_per_ep = seq_per_ep,
                                                            use_LSTM = use_LSTM,
                                                            use_stack = use_stack),
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
                                                        seq_per_ep = seq_per_ep,
                                                        use_LSTM = use_LSTM,
                                                        use_stack = use_stack),
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
                                                        seq_per_ep = seq_per_ep,
                                                        use_LSTM = use_LSTM,
                                                        use_stack = use_stack),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )


    n_train, n_val, n_test = len(train_loader)*batchsize, len(val_loader)*batchsize, len(test_loader)*batchsize

    print("modeltype----", model_type)
    if model_type == "CNN_stack_PR_FC":
        model = CNN_stack_PR_FC(num_channel=use_n_channels, cnn_fc_size = 1024 + use_n_im*2, num_output=predict_n_pr*2 )
    elif model_type == "CNN_PR_FC":
        model = CNN_PR_FC(cnn_fc_size = use_n_im*1026, num_output=predict_n_pr*2)
    elif model_type == "CNN_LSTM_encoder_decoder_images_PR":
        model = CNN_LSTM_encoder_decoder_images_PR(h_dim=2688, z_dim=1024, encoder_input_size = use_n_im*1026, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 2*predict_n_pr)
        #pretrained model
        CNN_part_tmp = AutoEncoder()
        CNN_part_tmp.load_state_dict(torch.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))
        model.encoder[0].weight = CNN_part_tmp.encoder[0].weight
        model.encoder[0].bias = CNN_part_tmp.encoder[0].bias
        model.encoder[3].weight = CNN_part_tmp.encoder[3].weight
        model.encoder[3].bias = CNN_part_tmp.encoder[3].bias
        model.encoder[6].weight = CNN_part_tmp.encoder[6].weight
        model.mu.weight = CNN_part_tmp.fc1.weight
        model.mu.bias = CNN_part_tmp.fc1.bias
        model.std.weight = CNN_part_tmp.fc2.weight
        model.std.bias = CNN_part_tmp.fc2.bias

    elif model_type == "LSTM_encoder_decoder_PR":
        model = LSTM_encoder_decoder_PR(h_dim=2688, z_dim=1024, encoder_input_size = use_n_im*2, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 2*predict_n_pr)
    elif model_type == "CNN_LSTM_encoder_decoder_images":
        model = CNN_LSTM_encoder_decoder_images(h_dim=2688, z_dim=1024, encoder_input_size = use_n_im*1024, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 2*predict_n_pr)
        #pretrained model
        CNN_part_tmp = AutoEncoder()
        CNN_part_tmp.load_state_dict(torch.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))
        model.encoder[0].weight = CNN_part_tmp.encoder[0].weight
        model.encoder[0].bias = CNN_part_tmp.encoder[0].bias
        model.encoder[3].weight = CNN_part_tmp.encoder[3].weight
        model.encoder[3].bias = CNN_part_tmp.encoder[3].bias
        model.encoder[6].weight = CNN_part_tmp.encoder[6].weight
        model.mu.weight = CNN_part_tmp.fc1.weight
        model.mu.bias = CNN_part_tmp.fc1.bias
        model.std.weight = CNN_part_tmp.fc2.weight
        model.std.bias = CNN_part_tmp.fc2.bias
    elif model_type == 'CNN_LSTM_decoder_images_PR':
        model = CNN_LSTM_decoder_images_PR(decoder_input_size = use_n_im*1026, decoder_hidden_size = 1000, output_size = 2*predict_n_pr)
        #pretrained model
        CNN_part_tmp = AutoEncoder()
        CNN_part_tmp.load_state_dict(torch.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))
        model.encoder[0].weight = CNN_part_tmp.encoder[0].weight
        model.encoder[0].bias = CNN_part_tmp.encoder[0].bias
        model.encoder[3].weight = CNN_part_tmp.encoder[3].weight
        model.encoder[3].bias = CNN_part_tmp.encoder[3].bias
        model.encoder[6].weight = CNN_part_tmp.encoder[6].weight
        model.mu.weight = CNN_part_tmp.fc1.weight
        model.mu.bias = CNN_part_tmp.fc1.bias
        model.std.weight = CNN_part_tmp.fc2.weight
        model.std.bias = CNN_part_tmp.fc2.bias
    else:
        raise ValueError("Model type not supported")

    if load_weight:
        model.load_state_dict(torch.load(weight_dir+"/" + model_type + "_predict_" + str(time_gap) + "_s_using_" + str(use_sec) + "_s_lr_" + str(learning_rate) + "_tmp.pth"))

    if cuda:
        model.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)

    # Loss functions
    loss_fn = nn.MSELoss(reduction = 'sum')

    best_val_error = np.inf
    best_train_error = np.inf

    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []


    model_weight = "/" + model_type + "_predict_" + str(time_gap) + "_s_using_" + str(use_sec) + "_s_lr_" + str(learning_rate) + "_tmp.pth"
    best_model_weight_path = weight_dir + model_weight
    tmp_str = '/' + model_type + "_predict_" + str(time_gap) + "_s_using_" + str(use_sec) + "_s_lr_" + str(learning_rate)

    plt.figure(1)
    fig = plt.figure(figsize=(22,15))
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='val loss')
    plt.legend(loc='upper right', fontsize=18)
    fig.canvas.draw()
    plt.title("Evolution of loss function")
    plt.xlabel("Epochs", fontsize = 18)
    plt.ylabel("Loss function", fontsize = 18)
    plt.show(block=False)


    # Finally, launch the training loop.
    start_time = time.time()


    print("Starting training...")
    # We iterate over epochs:

    for epoch in tqdm(range(num_epochs)):
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for k, data in enumerate(train_loader):
            # if k == 0:
                # print("CNN_p.state_dict() ",  model.state_dict())
            if use_LSTM:
                inputs, p_and_roll = data[0], data[1]
                if cuda:
                    inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll  = Variable(inputs), Variable(p_and_roll)

                loss = train(inputs, p_and_roll, model, optimizer, loss_fn, predict_n_pr, use_n_im)
                train_loss += loss

            else:
                inputs, p_and_roll, targets = data[0], data[1], data[2]
                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                optimizer.zero_grad()
                predictions = model(inputs, p_and_roll, use_n_im, cuda)
                loss = loss_fn(predictions, targets)/ predict_n_pr

                loss.backward()
                train_loss += loss.item()
                optimizer.step()


        train_error = (train_loss / n_train)*100

        # Do a full pass on validation data
        with th.no_grad():
            model.eval()
            for data in val_loader:
                if use_LSTM:
                    inputs, p_and_roll = data[0], data[1]
                    if cuda:
                        inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                    # Convert to pytorch variables
                    inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)

                    loss = eval(inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im)
                    val_loss += loss

                else:
                    inputs, p_and_roll, targets = data[0], data[1], data[2]
                    if cuda:
                        inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                    # Convert to pytorch variables
                    inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)


                    predictions = model(inputs, p_and_roll, use_n_im, cuda)
                    loss = loss_fn(predictions, targets)/ predict_n_pr

                    val_loss += loss.item()


            # Compute error per sample
            val_error = (val_loss / n_val)*100

            if val_error < best_val_error:
                best_val_error = val_error
                # Move back weights to cpu
                if cuda:
                    model.cpu()
                # Save Weights
                th.save(model.state_dict(), best_model_weight_path)

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

            json.dump(train_err_list, open(ress_dir+tmp_str+"_train_loss.json",'w'))
            json.dump(val_err_list, open(ress_dir+tmp_str+"_val_loss.json",'w'))
            print("  training loss:\t\t{:.6f}".format(train_error))
            print("  validation loss:\t\t{:.6f}".format(val_error))



    plt.savefig(img_dir+tmp_str +'_log_losses.png')
    plt.close()



    model.load_state_dict(th.load(best_model_weight_path))

    test_loss = 0.0

    with th.no_grad():

        origins = [{} for i in range(predict_n_pr)]
        origin_names = [lable_dir+ '/origin' + model_type +'_use_' + str(use_sec) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]
        preds = [{} for i in range(predict_n_pr)]
        pred_names = [lable_dir+'/pred' + model_type +'_use_' + str(use_sec) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]

        for key, data  in enumerate(test_loader):
            if use_LSTM:
                inputs, p_and_roll = data[0], data[1]
                if cuda:
                    inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)

                loss, origins, preds  = test(key, origins, preds , batchsize, inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im)
                test_loss += loss

            else:
                inputs, p_and_roll, targets = data[0], data[1], data[2]
                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)


                predictions = model(inputs, p_and_roll, use_n_im, cuda)

                key_tmp = np.linspace(key*batchsize , (key+1)*batchsize, batchsize, dtype =int )
                for pred_im in range(predict_n_pr):
                    tmp1 = gen_dict_for_json(key_tmp, targets[:,pred_im,:].cpu())
                    tmp2 = gen_dict_for_json(key_tmp, predictions[:,pred_im,:].cpu())

                    origins[pred_im] = {**origins[pred_im], **tmp1}
                    preds[pred_im] = {**preds[pred_im], **tmp2}

                loss = loss_fn(predictions, targets)/ predict_n_pr

                test_loss += loss.item()





        for i in range(predict_n_pr):
            json.dump(preds[i], open(pred_names[i],'w'))
            json.dump(origins[i], open(origin_names[i],'w'))

    test_error = (test_loss /n_test)*100

    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(min(val_err_list)))
    print("  test loss:\t\t\t{:.6f}".format(test_error))

    # write result into result.txt

    final_time = (time.time() - start_time)/60
    print("Total train time: {:.2f} mins".format(final_time))
    tmp2 = use_n_im
    if use_LSTM:
        tmp2 = LEN_SEQ


    write_result( ress_dir + "/result.txt", model_type, best_train_error, best_val_error,
                        test_error, time_gap, use_sec, seq_per_ep, tmp2, frame_interval, batchsize, seed, n_train, n_val,
                        n_test, num_epochs, [model], [optimizer], final_time)




    return best_train_error, best_val_error, test_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default= 50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default= 16, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--model_type', help='Model type: cnn', default="CNN_PR_FC", type=str, choices=['CNN_PR_FC', 'CNN_LSTM_encoder_decoder_images', 'LSTM_encoder_decoder_PR', 'CNN_stack_PR_FC', 'CNN_LSTM_encoder_decoder_images_PR', 'CNN_LSTM_decoder_images_PR'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-t', '--time_gap', help='Time gap', default= 12, type=int)
    parser.add_argument('-u', '--use_sec', help='How many seconds using for prediction ', default= 10, type=int)
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
            model_type=args.model_type,
            time_gap=args.time_gap,
            use_sec = args.use_sec )
    else:
        today = datetime.now()
        base_dir = "./Pre/results/BT_train_CNN_stack_PR_FC_"+str(today)
        os.mkdir(base_dir)
        xdata = []
        parm0_lr = []
        parm1_time_gap = []
        parm2_use_n_seconds_to_predict = []
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []
        time_gap_p = [15, 10, 5]
        lr_p = [ 5e-5, 1e-4, 5e-4]
        use_n_seconds_to_predict = [10, 8, 5]



        for lr in lr_p:
            to_plot = []
            plt.figure(lr*1000000)
            # resize pic to show details

            for ii in use_n_seconds_to_predict:
                for tg in time_gap_p:
                    tmp_train, tmp_val, tmp_test = main(

                                                        train_folder=args.train_folder,
                                                        num_epochs=args.num_epochs,
                                                        batchsize=args.batchsize,
                                                        learning_rate=lr,
                                                        cuda=args.cuda,
                                                        seed=args.seed,
                                                        load_weight = False,
                                                        model_type=args.model_type,
                                                        time_gap=tg,
                                                        use_sec = ii
                                                    )
                    parm0_lr.append(lr)
                    parm1_time_gap.append(tg)
                    parm2_use_n_seconds_to_predict.append(ii)
                    parm3_best_train_loss.append(tmp_train)
                    parm4_best_val_loss.append(tmp_val)
                    parm5_best_test_loss.append(tmp_test)

                to_plot.append(parm3_best_train_loss)




                for x in range(len(parm3_best_train_loss)):
                    print("---------------------------------------")
                    print("lr       - > ", parm0_lr[x])
                    print("time_gap - > ", parm1_time_gap[x])
                    print("number_im -> ", parm2_use_n_seconds_to_predict[x])
                    print("train     ->", parm3_best_train_loss[x])
                    print("val       ->", parm4_best_val_loss[x])
                    print("test      ->", parm5_best_test_loss[x])

                parm0_lr = []
                parm1_time_gap = []
                parm2_use_n_seconds_to_predict = []
                parm3_best_train_loss = []
                parm4_best_val_loss = []
                parm5_best_test_loss = []

            for ii in range(len(use_n_seconds_to_predict)):
                plt.plot(time_gap_p, to_plot[ii] , linewidth=1, alpha=0.9, label="test error using: " + str(use_n_seconds_to_predict[ii]) + "sec")

            plt.title("test_error - time_gap")
            plt.xlabel("Time_gap")
            plt.ylabel("Test_error")
            plt.legend(loc='upper right')
            plt.savefig(base_dir+'/error_test_lr_'+str(lr)+'.png')
            plt.close()
