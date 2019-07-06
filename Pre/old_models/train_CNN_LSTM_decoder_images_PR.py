"""
Train a neural network to predict vessel's movement
"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import json
import numpy as np
import torch as th

import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torchvision import transforms


from Pre.constants import RES_DIR, LEN_SEQ
from Pre.utils import loadOriginalLabels
from Pre.utils import JsonDatasetForLSTM as JsonDataset
from Pre.models import LSTM_decoder_simple2, AutoEncoder

# run this code under ssh mode, you need to add the following two lines codes.
# import matplotlib
# matplotlib.use('Agg')
"""if above line didn't work, use following two lines instead"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Pre.data_aug import imgTransform


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



def train(inputs, targets, CNN_p, LSTM_decoder, CNN_optimizer,
        LSTM_decoder_optimizer, criterion,
        time_gap=5, frame_interval= 12, sec_to_pred = 2):



    decoder_hidden = (LSTM_decoder.initHidden(targets.size(0)).cuda(),
                    LSTM_decoder.initHidden(targets.size(0)).cuda())

    CNN_optimizer.zero_grad()
    LSTM_decoder_optimizer.zero_grad()

    input_length = inputs.size(1)
    target_length = LEN_SEQ - int(24/frame_interval)*time_gap - int(24/frame_interval)*sec_to_pred

    loss = 0
    use_n_im = int(24/frame_interval)*sec_to_pred

    for im in range(use_n_im-1, target_length+use_n_im):

        features = [CNN_p(inputs[:,im-i,:,:,:])[0] for i in range(use_n_im-1, -1, -1)]
        PR = [targets[:,im-i,:] for i in range(use_n_im-1, -1, -1)]

        lstm_input_features = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = th.cat(lstm_input_features, 2).view(inputs.size(0), 1, -1)


        decoder_output, decoder_hidden = LSTM_decoder(lstm_input_features, decoder_hidden)

        im_to_predict = int(24/frame_interval)*time_gap
        decoder_output = decoder_output.view(inputs.size(0), im_to_predict, -1)
        loss += criterion(decoder_output, targets[:,im+1:im+im_to_predict+1,:])/im_to_predict


    loss.backward()

    CNN_optimizer.step()
    LSTM_decoder_optimizer.step()

    return loss.item() / target_length



def eval(inputs, targets, CNN_p, LSTM_decoder, criterion, time_gap=5, frame_interval= 12, sec_to_pred = 2):

    decoder_hidden = (LSTM_decoder.initHidden(targets.size(0)).cuda(),
                    LSTM_decoder.initHidden(targets.size(0)).cuda())


    input_length = inputs.size(1)
    target_length = LEN_SEQ - int(24/frame_interval)*time_gap - int(24/frame_interval)*sec_to_pred

    with th.no_grad():
        loss = 0
        use_n_im = int(24/frame_interval)*sec_to_pred

        for im in range(use_n_im-1, target_length+use_n_im):

            features = [CNN_p(inputs[:,im-i,:,:,:])[0] for i in range(use_n_im-1, -1, -1)]
            PR = [targets[:,im-i,:] for i in range(use_n_im-1, -1, -1)]

            lstm_input_features = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(use_n_im)]
            lstm_input_features = th.cat(lstm_input_features, 2).view(inputs.size(0), 1, -1)



            decoder_output, decoder_hidden = LSTM_decoder(lstm_input_features, decoder_hidden)

            im_to_predict = int(24/frame_interval)*time_gap
            decoder_output = decoder_output.view(inputs.size(0),im_to_predict, -1)
            loss += criterion(decoder_output, targets[:,im+1:im+im_to_predict+1,:])/im_to_predict

    return loss.item() / target_length


def test(i, origins, preds, batchsize, inputs, targets,
        CNN_p, LSTM_decoder,
        criterion, time_gap=5, frame_interval= 12, sec_to_pred = 2):

    decoder_hidden = (LSTM_decoder.initHidden(targets.size(0)).cuda(), LSTM_decoder.initHidden(targets.size(0)).cuda())


    target_length = LEN_SEQ - int(24/frame_interval)*time_gap - int(24/frame_interval)*sec_to_pred


    with th.no_grad():
        loss = 0
        use_n_im = int(24/frame_interval)*sec_to_pred
        for im in range(int(24/frame_interval)*sec_to_pred-1, target_length+int(24/frame_interval)*sec_to_pred):
            features = [CNN_p(inputs[:,im-i,:,:,:])[0] for i in range(use_n_im-1, -1, -1)]
            PR = [targets[:,im-i,:] for i in range(use_n_im-1, -1, -1)]

            lstm_input_features = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(use_n_im)]
            lstm_input_features = th.cat(lstm_input_features, 2).view(inputs.size(0), 1, -1)




            decoder_output, decoder_hidden = LSTM_decoder(lstm_input_features, decoder_hidden)

            im_to_predict = int(24/frame_interval)*time_gap
            decoder_output = decoder_output.view(inputs.size(0), im_to_predict, -1)
            loss += criterion(decoder_output, targets[:,im+1:im+im_to_predict+1,:])/im_to_predict

            key_tmp = np.linspace(i*target_length*batchsize + (im-3)*batchsize , i*target_length*batchsize + (im-2)*batchsize - 1, batchsize, dtype =int )
            for pred_im in range(im_to_predict):
                tmp1 = gen_dict_for_json(key_tmp, targets[:,im+pred_im+1,:].cpu())
                tmp2 = gen_dict_for_json(key_tmp, decoder_output[:,pred_im,:].cpu())

                origins[pred_im] = {**origins[pred_im], **tmp1}
                preds[pred_im] = {**preds[pred_im], **tmp2}

    return loss.item() / target_length, origins, preds


def main(train_folder, num_epochs=100, batchsize=32, learning_rate=0.001, seed=42,
        cuda=True, random_trans=0.0, evaluate_print=1,
        time_gap=5, num_images = 1, frame_interval = 12, sec_to_pred = 2):



    print('has cuda?', cuda)
    print('BS--- ', batchsize)
    # Will be changed to separe plus efective
    train_labels, val_labels, test_labels = loadOriginalLabels(train_folder, 1, 320, p_train=0.7, p_val=0.15, p_test=0.15)

    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)

    # Retrieve number of samples per set

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
                                            shuffle=False,
                                            **kwargs)

    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_loader), len(val_loader), len(test_loader)

    im_to_predict = int(24/frame_interval)*time_gap

    # LSTM_encoder_p = LSTM_encoder(input_size=sec_to_pred*int(24/frame_interval)*1026, hidden_size=1024, num_layers=1)
    LSTM_decoder_total = LSTM_decoder_simple2(hidden_size=sec_to_pred*int(24/frame_interval)*1026, output_size = 2*im_to_predict, num_layers=1)
    CNN_p = AutoEncoder()
    CNN_p.load_state_dict(torch.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))

    # Freeze model weights
    # for param in CNN_p.parameters():
    #     param.requires_grad = False

    if cuda:
        # LSTM_encoder_p.cuda()
        LSTM_decoder_total.cuda()
        CNN_p.cuda()
    # L2 penalty
    weight_decay = 1e-3
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loss and optimizer
    criteration = nn.MSELoss(reduction = 'sum')#nn.NLLLoss()

    CNN_optimizer = th.optim.Adam(CNN_p.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # LSTM_encoder_optimizer = th.optim.Adam(LSTM_encoder_p.parameters(), lr=learning_rate, weight_decay=weight_decay)
    LSTM_decoder_optimizer = th.optim.Adam(LSTM_decoder_total.parameters(), lr=learning_rate, weight_decay=weight_decay)


    best_error = np.inf
    best_train_error = np.inf
    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []

    file_CNN_model = "CNN_LSTM_simple_2_CNN_part_{}s_{}_s_to_pred_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    # file_encoder_model = "CNN_LSTM_simple_2_Encoder_part_{}s_{}_s_to_pred_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)
    file_decoder_model = "CNN_LSTM_simple_2_Decoder_part_{}s_{}_s_to_pred_lr_{}_tmp.pth".format(time_gap, sec_to_pred, learning_rate)

    best_model_CNN_part = RES_DIR + file_CNN_model
    # best_model_encoder_part = RES_DIR + file_encoder_model
    best_model_decoder_part = RES_DIR + file_decoder_model

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
        # LSTM_encoder_p.train()
        LSTM_decoder_total.train()
        CNN_p.train()
        train_loss, val_loss = 0.0, 0.0
        # Full pass on training data
        # Update the model after each minibatch
        for i, (inputs, targets) in enumerate(train_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)
            loss = train(inputs, targets, CNN_p, LSTM_decoder_total, CNN_optimizer,
                                 LSTM_decoder_optimizer, criteration, time_gap = time_gap, sec_to_pred = sec_to_pred)
            train_loss += loss

        train_l = train_loss / (n_train*batchsize)
        # LSTM_encoder_p.eval()
        LSTM_decoder_total.eval()
        CNN_p.eval()

        for i, (inputs, targets) in enumerate(val_loader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, targets = Variable(inputs), Variable(targets)
            loss = eval(inputs, targets, CNN_p, LSTM_decoder_total, criteration,
                        time_gap = time_gap, sec_to_pred = sec_to_pred)
            val_loss += loss

        val_l = val_loss/ (n_val*batchsize)

        if val_l < best_error:
            best_error = val_l
            # Move back weights to cpu
            if cuda:
                # LSTM_encoder_p.cpu()
                LSTM_decoder_total.cpu()
                CNN_p.cpu()

            # Save Weights of all parts
            th.save(CNN_p.state_dict(), best_model_CNN_part)
            # th.save(LSTM_encoder_p.state_dict(), best_model_encoder_part)
            th.save(LSTM_decoder_total.state_dict(), best_model_decoder_part)

            if cuda:
                # LSTM_encoder_p.cuda()
                LSTM_decoder_total.cuda()
                CNN_p.cuda()

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

            print("  training loss:\t\t{:.6f}".format(train_l))
            print("  validation loss:\t\t{:.6f}".format(val_l))


    plt.savefig(RES_DIR+'CNN_LSTM_simple_2_model_'+'_predict_'+str(time_gap)
                            +'s_using_'+str(sec_to_pred)+'_lr_'+str(learning_rate)+'s_loss'+'.png')
    # LOQD the best model
    CNN_p.load_state_dict(th.load(best_model_CNN_part))
    # LSTM_encoder_p.load_state_dict(th.load(best_model_encoder_part))
    LSTM_decoder_total.load_state_dict(th.load(best_model_decoder_part))

    # After training, we compute and print the test error:
    print('Test starting...')
    test_loss = 0.0

    origins = [{} for i in range(im_to_predict)]
    origin_names = [RES_DIR+'cnn_lstm_simple_2_origin_label_use_'+str(sec_to_pred)+'_to_predict_'+str(i+1)+':'+str(im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(im_to_predict)]
    preds = [{} for i in range(im_to_predict)]
    pred_names = [RES_DIR+'cnn_lstm_simple_2_pred_label_use_'+str(sec_to_pred)+'_to_predict_'+str(i+1)+':'+str(im_to_predict)+'_lr_'+str(learning_rate)+'_.json' for i in range(im_to_predict)]


    for i , (inputs, targets) in enumerate(test_loader):

        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)


        loss, origins, preds  = test(i, origins, preds, batchsize, inputs,
                                    targets, CNN_p, LSTM_decoder_total,
                                    criteration, time_gap = time_gap, sec_to_pred= sec_to_pred)
        test_loss += loss

    test_l = test_loss / (n_test*batchsize)
    for i in range(im_to_predict):
        json.dump(preds[i], open(pred_names[i],'w'))
        json.dump(origins[i], open(origin_names[i],'w'))

    print("Final results:")
    print("  best train loss:\t\t{:.6f}".format(best_train_error))
    print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  test loss:\t\t\t{:.6f}".format(test_l))

    # write result into result_CNN_LSTM.txt
    # format fixed because we use this file later in pltModelTimegap.py
    with open("./Pre/result_CNN_LSTM_simple_2.txt", "a") as f:
        f.write("current model: ")
        f.write("CNN_LSTM")
        f.write("\nbest train error:")
        f.write(str(best_train_error))
        f.write("\nbest validation loss:")
        f.write(str(best_error))
        f.write("\nfinal test loss:")
        f.write(str(test_l))
        f.write("\n")
        f.write("time to predict is (seconds):")
        f.write(str(time_gap))
        f.write("\n")
        f.write("Time use to predict:")
        f.write(str(sec_to_pred))
        f.write("\n")
        f.write("Train examples (images):")
        f.write(str(n_train*batchsize*LEN_SEQ))
        f.write("\n")
        f.write("Val examples (images):")
        f.write(str(n_val*batchsize*LEN_SEQ))
        f.write("\n")
        f.write("Test examples (images):")
        f.write(str(n_test*batchsize*LEN_SEQ))
        f.write("BATCH_SIZE:")
        f.write(str(batchsize))
        f.write("\n")
        f.write(str(CNN_p))
        f.write("\n")
        f.write(str(LSTM_decoder_total))
        f.write("\n")
        f.write(str(CNN_optimizer))
        f.write("\n")
        f.write(str(LSTM_decoder_optimizer))
        f.write("\n\n")
    f.close()
    print("Total train time: {:.2f} mins".format((time.time() - start_time)/60))
    return best_train_error, best_error, test_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=64, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=5e-4, type=float)
    parser.add_argument('-t', '--time_gap', help='Time in seconds to predict ', default=5, type=int)
    parser.add_argument('-u', '--sec_to_pred', help='How many seconds take to predict smth ', default=5, type=int)
    parser.add_argument('-ni', '--num_images', help='Number of images in the series', default=1, type=int)
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
            time_gap=args.time_gap,
            sec_to_pred = args.sec_to_pred)
    else:
        parm1_lr = []
        parm2_sec_to_pred = []
        parm22_pred_s =[]
        parm3_best_train_loss = []
        parm4_best_val_loss = []
        parm5_best_test_loss = []

        sec_to_pred_p = [10, 8, 6]
        pred_p = [10, 15]
        lr_p = [5e-4]

        for pred_t in pred_p:
            for stp in sec_to_pred_p:
                for lr in lr_p:
                    tmp_train, tmp_val, tmp_test = main(train_folder=args.train_folder,
                                                        num_epochs=args.num_epochs,
                                                        batchsize=args.batchsize,
                                                        learning_rate=lr,
                                                        cuda=args.cuda,
                                                        seed=args.seed,
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
