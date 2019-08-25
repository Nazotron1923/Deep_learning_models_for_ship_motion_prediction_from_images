"""
Train a neural network to predict vessel's movement
"""
from __future__ import absolute_import

import argparse
import time
import json
import os
from datetime import datetime
import random
import numpy as np
import torch as th
from tqdm import tqdm

import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from Pre.constants import SEQ_PER_EPISODE_C, LEN_SEQ, RES_DIR
from Pre.utils import loadLabels, gen_dict_for_json, write_result, use_pretrainted
from Pre.utils import JsonDataset_universal as JsonDataset

from Pre.earlyStopping import EarlyStopping
from Pre.models import CNN_stack_FC_first, CNN_stack_FC, CNN_stack_PR_FC, CNN_LSTM_encoder_decoder_images_PR, AutoEncoder, LSTM_encoder_decoder_PR, CNN_LSTM_encoder_decoder_images, CNN_LSTM_decoder_images_PR, CNN_PR_FC, CNN_LSTM_image_encoder_PR_encoder_decoder

from Pre.get_hyperparameters_configuration import get_params
from Pre.hyperband import Hyperband
import matplotlib.pyplot as plt
plt.switch_backend('agg')


"""
If you use model that contain an LSTM architecture, this function implements
training throughout the sequence (step by step)
"""
def train(cuda, change_fps, inputs, targets, model, optimizer, criterion, predict_n_pr, use_n_im, use_2_encoders = False):
    """
    Args:
        cuda (boolean): If True use cuda device
        inputs (tensor): The sequence of frames used for training
        targets (tensor): Pitch and roll for each image
        model (torch.nn.Module): the model which will be used for training
        optimizer (torch.optim): the optimizer which will be used for optimize the model
        criterion (torch.nn.modules.loss): type of loss function
        predict_n_pr (int): the number of frames for which the pitch and roll will be predicted
        use_n_im (int): the number of frames we use to predict pitch and roll
        use_2_encoders (boolean) : If True we use the model with two encoders (for frames and for pitch & roll)
                                Default: False
    """

    if cuda:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                        model.initHiddenEncoder(inputs.size(0)).cuda())
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)).cuda(),
                        model.initHiddenEncoderIm(inputs.size(0)).cuda())
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)).cuda(),
                        model.initHiddenEncoderPR(inputs.size(0)).cuda())

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                        model.initHiddenDecoder(targets.size(0)).cuda())
    else:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)),
                        model.initHiddenEncoder(inputs.size(0)))
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)),
                        model.initHiddenEncoderIm(inputs.size(0)))
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)),
                        model.initHiddenEncoderPR(inputs.size(0)))

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)),
                        model.initHiddenDecoder(targets.size(0)))

    # How many steps will the algorithm take through the sequence
    tmp = LEN_SEQ
    if change_fps:
        tmp = int(LEN_SEQ/2)
    target_length = tmp - predict_n_pr - use_n_im

    optimizer.zero_grad()
    loss = 0

    for im in range(use_n_im-1, target_length+use_n_im):
        # Get one input for model
        image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
        pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

        # Prediction
        if not use_2_encoders:
            prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)
        else:
            prediction, im_encoder_hidden, pr_encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden)

        # calculate loss function
        loss += criterion(prediction, targets[:,im+1 : im+predict_n_pr+1,:])/predict_n_pr


    loss.backward()
    optimizer.step()

    return loss.item() / target_length


"""
If you use model that contain an LSTM architecture, this function implements
evaluation throughout the sequence (step by step)
"""
def eval(cuda, change_fps, inputs, targets, model, criterion, predict_n_pr, use_n_im, use_2_encoders = False):
    """
    Args:
        cuda (boolean): If True use cuda device
        inputs (tensor): The sequence of frames used for training
        targets (tensor): Pitch and roll for each image
        model (torch.nn.Module): the model which will be used for training
        criterion (torch.nn.modules.loss): type of loss function
        predict_n_pr (int): the number of frames for which the pitch and roll will be predicted
        use_n_im (int): the number of frames we use to predict pitch and roll
        use_2_encoders (boolean) : If True we use the model with two encoders (for frames and for pitch & roll)
                                Default: False
    """
    if cuda:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                        model.initHiddenEncoder(inputs.size(0)).cuda())
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)).cuda(),
                        model.initHiddenEncoderIm(inputs.size(0)).cuda())
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)).cuda(),
                        model.initHiddenEncoderPR(inputs.size(0)).cuda())

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                        model.initHiddenDecoder(targets.size(0)).cuda())
    else:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)),
                        model.initHiddenEncoder(inputs.size(0)))
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)),
                        model.initHiddenEncoderIm(inputs.size(0)))
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)),
                        model.initHiddenEncoderPR(inputs.size(0)))

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)),
                        model.initHiddenDecoder(targets.size(0)))

    # How many steps will the algorithm take through the sequence
    tmp = LEN_SEQ
    if change_fps:
        tmp = int(LEN_SEQ/2)
    target_length = tmp - predict_n_pr - use_n_im
    # For evaluation we don't need a gradient
    with th.no_grad():
        loss = 0

        for im in range(use_n_im-1, target_length+use_n_im):
            # Get one input for model
            image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
            pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

            # Prediction
            if not use_2_encoders:
                prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)
            else:
                prediction, im_encoder_hidden, pr_encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden)

            # Calculate loss function
            loss += criterion(prediction, targets[:,im+1:im+predict_n_pr+1,:])/predict_n_pr

    return loss.item() / target_length


"""
If you use model that contain an LSTM architecture, this function implements
testing throughout the sequence (step by step) and will save results in dictionary for visualization
"""
def test(cuda, change_fps, i, origins, preds, batchsize, inputs, targets,
        model, criterion, predict_n_pr, use_n_im, use_2_encoders = False):

    """
    Args:
        cuda (boolean): If True use cuda device
        i (int):    index for writing in right order to dictionary
        origins (dict): a dictionary that stores the original pitch and roll values
        preds (dict) : a dictionary that stores the predicted pitch and roll values
        batchsize (int): batchsize
        inputs (tensor): The sequence of frames used for training
        targets (tensor): Pitch and roll for each image
        model (torch.nn.Module): the model which will be used for training
        criterion (torch.nn.modules.loss): type of loss function
        predict_n_pr (int): the number of frames for which the pitch and roll will be predicted
        use_n_im (int): the number of frames we use to predict pitch and roll
        use_2_encoders (boolean) : If True we use the model with two encoders (for frames and for pitch & roll)
                                Default: False
    """

    if cuda:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)).cuda(),
                        model.initHiddenEncoder(inputs.size(0)).cuda())
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)).cuda(),
                        model.initHiddenEncoderIm(inputs.size(0)).cuda())
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)).cuda(),
                        model.initHiddenEncoderPR(inputs.size(0)).cuda())

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)).cuda(),
                        model.initHiddenDecoder(targets.size(0)).cuda())
    else:
        # Preparation of hidden vectors for LSTM encoders and decoders
        if not use_2_encoders:
            encoder_hidden = (model.initHiddenEncoder(inputs.size(0)),
                        model.initHiddenEncoder(inputs.size(0)))
        else:
            im_encoder_hidden = (model.initHiddenEncoderIm(inputs.size(0)),
                        model.initHiddenEncoderIm(inputs.size(0)))
            pr_encoder_hidden = (model.initHiddenEncoderPR(inputs.size(0)),
                        model.initHiddenEncoderPR(inputs.size(0)))

        decoder_hidden = (model.initHiddenDecoder(targets.size(0)),
                        model.initHiddenDecoder(targets.size(0)))

    # How many steps will the algorithm take through the sequence
    tmp = LEN_SEQ
    if change_fps:
        tmp = int(LEN_SEQ/2)
    target_length = tmp - predict_n_pr - use_n_im
    # For testing we don't need a gradient
    with th.no_grad():
        loss = 0

        for im in range(use_n_im-1, target_length+use_n_im):
            # Get one input for model
            image_s = [inputs[:,im-i,:,:,:] for i in range(use_n_im - 1, -1, -1)]
            pr_s = [targets[:,im-i,:] for i in range(use_n_im - 1, -1, -1)]

            # Prediction
            if not use_2_encoders:
                prediction, encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden)
            else:
                prediction, im_encoder_hidden, pr_encoder_hidden, decoder_hidden = model(image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden)

            # Calculate loss function
            loss += criterion(prediction, targets[:,im+1:im+predict_n_pr+1,:])/predict_n_pr

            # Create new index for save results
            key_tmp = np.linspace(i*target_length*batchsize + (im-use_n_im+1)*batchsize , i*target_length*batchsize + (im-use_n_im+2)*batchsize - 1, batchsize, dtype =int )

            # For each frame which we will predict in sequence save the result
            for pred_im in range(predict_n_pr):
                tmp1 = gen_dict_for_json(key_tmp, targets[:,im+pred_im+1,:].cpu())
                tmp2 = gen_dict_for_json(key_tmp, prediction[:,pred_im,:].cpu())

                origins[pred_im] = {**origins[pred_im], **tmp1}
                preds[pred_im] = {**preds[pred_im], **tmp2}

    return loss.item() / target_length, origins, preds


def main(args, num_epochs = 30):
    """
    Args:
        args (dict):    Dictionary of parametres
            args['train_folder']    (str): folder's prefix where dataset is stored
            args['batchsize']       (int): batchsize
            args['opt']             (str): optimizer type
            args['learning_rate']   (float): learning_rate
            args['seed']            (int): number to fix random processes
            args['cuda']            (boolean): True if we can use GPU
            args['load_weight']     (boolean): True if we will load model
            args['load_weight_date'](str): date of the test (part of the path)
            args['model_type']      (str): model type
            args['encoder_latent_vector'] (int): size of encoder latent vector
            args['decoder_latent_vector'] (int): size of decoder latent vector
            args['future_window_size']         (int): number of seconds to predict
            args['past_window_size']          (int): number of seconds using like input
            args['frame_interval']   (int): interval at witch the data was generated
            args["weight_decay"]     (float): L2 penalty
            args["use_n_episodes"]   (int): number of episodes use for work
            args["test_dir"]         (str): if you run a parameter test, all results will be stored in test folder

        num_epochs (int) : Number of epochs
                            Default: 30
    Return (dict) : {float, float, float, bool}
                    best train loss, best validation loss, final test loss, early stops
    """
    # setting parametres for training

    train_folder = args['train_folder']        # 50
    batchsize = args['batchsize']            # 32
    opt = args['opt']
    learning_rate = args['learning_rate']    # 0.0001
    seed = args['seed']                      # 42
    cuda = args['cuda']                      # True
    load_weight = args['load_weight']        # False,
    load_weight_date = args['load_weight_date']
    model_type = args['model_type']          # "CNN_LSTM_encoder_decoder_images_PR"
    encoder_latent_vector = args['encoder_latent_vector']
    decoder_latent_vector = args['decoder_latent_vector']
    evaluate_print = 1
    future_window_size = args['future_window_size']              # 5
    past_window_size = args['past_window_size']                # 5,
    frame_interval = args['frame_interval']  # 12
    weight_decay = args["weight_decay"]      # 1e-3
    use_n_episodes = args["use_n_episodes"]    # 540
    test_dir = args["test_dir"]
    change_fps = args["change_fps"]
    print(args)

    im_in_one_second = int(24/frame_interval)

    predict_n_pr = im_in_one_second*future_window_size
    use_n_im = im_in_one_second*past_window_size

    use_LSTM = False
    use_stack = False
    use_n_channels = 3
    seq_per_ep = SEQ_PER_EPISODE_C
    use_2_encoders = False
    # parametr for different models
    if 'LSTM' in model_type:
        use_LSTM = True
    else:
        seq_per_ep = int(360/use_n_im)

    if 'stack' in model_type:
        use_stack = True
        use_n_channels = 3*use_n_im

    # Set early stopping technique
    early_stopping = EarlyStopping(patience=8, verbose=False)

    # indicate randomseed , so that we will be able to reproduce the result in the future
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are using GPU
    print('Use cuda ->  ', cuda)
    if cuda:
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.enabled = False
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    #---------------------------------------------------------------------------

    # Load model if True
    if load_weight:
        today = load_weight_date
    else:
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create folders for results
    base_dir = "./Pre/results"+test_dir+"/train_"+ model_type +"_using_" +str(past_window_size)+  "_s_to_predict_"+str(future_window_size)+ "_s_lr_" + str(learning_rate) + "_" + today
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


    # parametres for different models

    tmp_str = '/' + model_type + "_predict_" + str(future_window_size) + "_s_using_" + str(past_window_size) + "_s_lr_" + str(learning_rate)
    best_model_weight_path = weight_dir + tmp_str + "_tmp.pth"

    # split our dataset for three parts
    train_labels, val_labels, test_labels = loadLabels(train_folder, 0, use_n_episodes, seq_per_ep, p_train=0.7, p_val=0.15, p_test=0.15)

    print('len(train_labels) -> ', len(train_labels))
    print('len(val_labels) -> ', len(val_labels))
    print('len(test_labels) -> ', len(test_labels))
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
                                                            use_stack = use_stack,
                                                            change_fps = change_fps),
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
                                                        use_stack = use_stack,
                                                        change_fps = change_fps),
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
                                                        use_stack = use_stack,
                                                        change_fps = change_fps),
                                            batch_size=batchsize,
                                            shuffle=True,
                                            **kwargs
                                        )


    n_train, n_val, n_test = len(train_loader)*batchsize, len(val_loader)*batchsize, len(test_loader)*batchsize

    if change_fps:
        predict_n_pr = future_window_size
        use_n_im = past_window_size

    print("Model  --->  ", model_type)
    if model_type == "CNN_stack_PR_FC":
        model = CNN_stack_PR_FC(cuda = cuda, num_channel=use_n_channels, cnn_fc_size = 1024 + use_n_im*2, num_output=predict_n_pr*2 )
    elif model_type == "CNN_PR_FC":
        model = CNN_PR_FC(cuda = cuda, cnn_fc_size = use_n_im*1026, num_output=predict_n_pr*2)
    elif model_type == "CNN_stack_FC_first":
        model = CNN_stack_FC_first(cuda = cuda, num_channel = use_n_channels,  cnn_fc_size = 1024, num_output=predict_n_pr*2)
    elif model_type == "CNN_stack_FC":
        model = CNN_stack_FC(cuda = cuda, num_channel = use_n_channels,  cnn_fc_size = 1024, num_output=predict_n_pr*2)
    elif model_type == "CNN_LSTM_encoder_decoder_images_PR":
        model = CNN_LSTM_encoder_decoder_images_PR(cuda = cuda, encoder_input_size = use_n_im*1026, encoder_hidden_size = encoder_latent_vector, decoder_input_size = encoder_latent_vector, decoder_hidden_size = decoder_latent_vector,  output_size = 2*predict_n_pr)
        #use pretrained model
        use_pretrainted(model, AutoEncoder())
    elif model_type == "LSTM_encoder_decoder_PR":
        model = LSTM_encoder_decoder_PR(cuda = cuda, encoder_input_size = use_n_im*2, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 2*predict_n_pr)
    elif model_type == "CNN_LSTM_encoder_decoder_images":
        model = CNN_LSTM_encoder_decoder_images(cuda = cuda, encoder_input_size = use_n_im*1024, encoder_hidden_size = encoder_latent_vector, decoder_hidden_size = encoder_latent_vector,  output_size = 2*predict_n_pr)
        #use pretrained model
        use_pretrainted(model, AutoEncoder())
    elif model_type == 'CNN_LSTM_decoder_images_PR':
        model = CNN_LSTM_decoder_images_PR(cuda = cuda, decoder_input_size = use_n_im*1026, decoder_hidden_size = 1000, output_size = 2*predict_n_pr)
        #use pretrained model
        CNN_part_tmp = AutoEncoder()
        use_pretrainted(model, AutoEncoder())
    elif model_type == "CNN_LSTM_image_encoder_PR_encoder_decoder":
        model = CNN_LSTM_image_encoder_PR_encoder_decoder(cuda = cuda, im_encoder_input_size = use_n_im*1024, pr_encoder_input_size = use_n_im*2 , im_encoder_hidden_size = 600, pr_encoder_hidden_size = 300, decoder_hidden_size = 900,  output_size = predict_n_pr*2)
        #use pretrained model
        use_pretrainted(model, AutoEncoder())
        use_2_encoders = True
    else:
        raise ValueError("Model type not supported")

    # Load model's weights
    if load_weight:
        model.load_state_dict(torch.load(weight_dir+"/" + model_type + "_predict_" + str(future_window_size) + "_s_using_" + str(past_window_size) + "_s_lr_" + str(learning_rate) + "_tmp.pth"))

    # Move model to the GPU if possible
    if cuda:
        model.cuda()

    # Optimizers
    if opt == "adam":
        optimizer = th.optim.Adam(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay
                                )
    elif opt == "sgd":
        optimizer = th.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=weight_decay,
                                    nesterov=True
                                )

    # Loss functions
    loss_fn = nn.MSELoss(reduction = 'sum')

    best_val_error = np.inf
    best_train_loss = np.inf

    # error list for updata loss figure
    train_err_list = []
    val_err_list = []
    # epoch list
    xdata = []

    plt.figure(1)
    fig = plt.figure(figsize=(22,15))
    ax = fig.add_subplot(111)
    li, = ax.plot(xdata, train_err_list, 'b-', label='train loss')
    l2, = ax.plot(xdata, val_err_list, 'r-', label='validation loss')
    plt.legend(loc='upper right', fontsize=18)
    fig.canvas.draw()
    plt.title("Evolution of loss function")
    plt.xlabel("Epochs", fontsize = 18)
    plt.ylabel("Loss function", fontsize = 18)
    plt.show(block=False)


    # Finally, launch the training loop.
    start_time = time.time()
    print("Start training...")

    # We iterate over epochs:
    for epoch in tqdm(range(num_epochs)):
        # Do a full pass on training data
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for k, data in enumerate(train_loader):
            # use right training process for different models
            if use_LSTM:
                # unpacked data
                inputs, p_and_roll = data[0], data[1]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll  = Variable(inputs), Variable(p_and_roll)
                # training through the sequence
                loss = train(cuda, change_fps, inputs, p_and_roll, model, optimizer, loss_fn, predict_n_pr, use_n_im, use_2_encoders)
                train_loss += loss

            else:
                # unpacked data
                inputs, p_and_roll, targets = data[0], data[1], data[2]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                # training process
                optimizer.zero_grad()
                predictions = model(inputs, p_and_roll, use_n_im)
                loss = loss_fn(predictions, targets) / predict_n_pr

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

        # train MSE for pitch and roll
        train_error = train_loss / n_train

        # if the train_error is so big stop training and save time (usefull for Hyperband)
        if np.isnan(train_error):
            return {"best_train_loss": np.inf, "best_val_loss": np.inf, "final_test_loss": np.inf}

        # Do a full pass on validation data
        with th.no_grad():
            # Switch to evaluation mode
            model.eval()
            for data in val_loader:
                # use right validation process for different models
                if use_LSTM:
                    # unpacked data
                    inputs, p_and_roll = data[0], data[1]
                    # move data to GPU
                    if cuda:
                        inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                    # Convert to pytorch variables
                    inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)
                    # validation through the sequence
                    loss = eval(cuda, change_fps, inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im, use_2_encoders)
                    val_loss += loss

                else:
                    # unpacked data
                    inputs, p_and_roll, targets = data[0], data[1], data[2]
                    # move data to GPU
                    if cuda:
                        inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                    # Convert to pytorch variables
                    inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                    # validation process
                    predictions = model(inputs, p_and_roll, use_n_im)
                    loss = loss_fn(predictions, targets)/ predict_n_pr
                    val_loss += loss.item()

            # validation MSE for pitch and roll
            val_error = val_loss / n_val

            early_stopping(val_error, model, best_model_weight_path, cuda)

            if train_error < best_train_loss:
                best_train_loss = train_error

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

            # Save evolution of train and validation loss functions
            json.dump(train_err_list, open(ress_dir+tmp_str+"_train_loss.json",'w'))
            json.dump(val_err_list, open(ress_dir+tmp_str+"_val_loss.json",'w'))
            print("  training avg loss [normalized -1, 1]   :\t\t{:.6f}".format(train_error))
            print("  validation avg loss [normalized -1, 1] :\t\t{:.6f}".format(val_error))

        if early_stopping.early_stop:
            print("----Early stopping----")
            break

    # Save figure
    plt.savefig(img_dir+tmp_str +'_log_losses.png')
    plt.close()

    # Load the best weight configuration
    model.load_state_dict(th.load(best_model_weight_path))

    print("Start testing...")
    test_loss = 0.0

    with th.no_grad():

        # Preparation files for saving origin and predicted pitch and roll for visualization
        origins = [{} for i in range(predict_n_pr)]
        origin_names = [lable_dir+ '/origin' + model_type +'_use_' + str(past_window_size) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]
        preds = [{} for i in range(predict_n_pr)]
        pred_names = [lable_dir+'/pred' + model_type +'_use_' + str(past_window_size) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]

        for key, data  in enumerate(test_loader):
            # use right testing process for different models
            if use_LSTM:
                # unpacked data
                inputs, p_and_roll = data[0], data[1]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)
                # test through the sequence
                loss, origins, preds  = test(cuda, change_fps, key, origins, preds , batchsize, inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im, use_2_encoders)
                test_loss += loss

            else:
                # unpacked data
                inputs, p_and_roll, targets = data[0], data[1], data[2]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                predictions = model(inputs, p_and_roll, use_n_im)

                # save results of prediction for visualization
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

    final_test_loss = test_loss /n_test

    print("Final results:")
    print("  best avg training loss [normalized (-1 : 1) ]:\t\t{:.6f}".format(best_train_loss))
    print("  best avg validation loss [normalized (-1 : 1) ]:\t\t{:.6f}".format(min(val_err_list)))
    print("  test avg loss[normalized (-1 : 1) ]:\t\t\t{:.6f}".format(final_test_loss))

    # write result into result.txt
    final_time = (time.time() - start_time)/60
    print("Total train time: {:.2f} mins".format(final_time))

    # set lenght of sequence used
    tmp_seq_len = use_n_im
    if use_LSTM:
        tmp_seq_len = LEN_SEQ

    # write configuration in file
    write_result(args, [model], [optimizer], result_file_name = ress_dir + "/result.txt",
                best_train_loss = best_train_loss, best_val_loss = early_stopping.val_loss_min,
                final_test_loss = final_test_loss, time = final_time, seq_per_ep = seq_per_ep,
                seq_len = tmp_seq_len, num_epochs = num_epochs
                )
    return {"best_train_loss": best_train_loss, "best_val_loss": early_stopping.val_loss_min, "final_test_loss": final_test_loss, "early_stop": early_stopping.early_stop}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('-tf', '--train_folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default= 50, type=int)
    parser.add_argument('--batchsize', help='Batch size', default= 24, type=int)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0001, type=float)
    parser.add_argument('--opt', help='Choose optimizer: cnn', default="adam", type=str, choices=['adam', 'sgd'])
    parser.add_argument('--test_dir', help='if test of hyperparametres ', default="", type=str)

    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no_cuda', action='store_true', default=True, help='Disables CUDA training')

    parser.add_argument('--load_model', action='store_true', default=False, help='LOAD_MODEL (to continue training)')
    parser.add_argument('--load_weight_date', help='Enter test date', default="2019-07-05 00:36", type=str)

    parser.add_argument('--model_type', help='Model type: cnn', default="LSTM_encoder_decoder_PR", type=str, choices=['CNN_stack_FC_first', 'CNN_stack_FC', 'CNN_LSTM_image_encoder_PR_encoder_decoder', 'CNN_PR_FC', 'CNN_LSTM_encoder_decoder_images', 'LSTM_encoder_decoder_PR', 'CNN_stack_PR_FC', 'CNN_LSTM_encoder_decoder_images_PR', 'CNN_LSTM_decoder_images_PR'])
    parser.add_argument('--encoder_latent_vector', help='Size of encoder-latent vector for LSTM', default= 300, type=int)
    parser.add_argument('--decoder_latent_vector', help='Size of decoder-latent vector for LSTM', default= 300, type=int)

    parser.add_argument('-fws', '--future_window_size', help='Time (seconds) to predict', default= 12, type=int)
    parser.add_argument('-pws', '--past_window_size', help='How many seconds using for prediction ', default= 10, type=int)
    parser.add_argument('--frame_interval', help='frame_interval which used for data generetion ', default= 12, type=int)
    parser.add_argument('-wd', '--weight_decay', help='Weight_decay', default=0.001, type=float)
    parser.add_argument('--use_n_episodes', help='How many episodes use as dataset ', default= 540, type=int)

    parser.add_argument('--change_fps', action='store_true', default=False, help='change_fps')
    parser.add_argument('--test', help='Test hyperparameters', default=0, type=int)
    args = parser.parse_args()

    args.cuda = (not args.no_cuda) and th.cuda.is_available()

    if args.test == 0:
        hyperparams = {}
        hyperparams['train_folder'] = args.train_folder
        hyperparams['batchsize'] = args.batchsize
        hyperparams['learning_rate'] = args.learning_rate
        hyperparams['opt'] = args.opt
        hyperparams['seed'] = args.seed
        hyperparams['cuda'] = args.cuda
        hyperparams['load_weight'] = args.load_model
        hyperparams['load_weight_date'] = args.load_weight_date
        hyperparams['model_type'] = args.model_type
        hyperparams['encoder_latent_vector'] = args.encoder_latent_vector
        hyperparams['decoder_latent_vector'] = args.decoder_latent_vector
        hyperparams['future_window_size'] = args.future_window_size
        hyperparams['past_window_size'] = args.past_window_size
        hyperparams['frame_interval'] = args.frame_interval
        hyperparams["weight_decay"] = args.weight_decay
        hyperparams["use_n_episodes"] = args.use_n_episodes
        hyperparams['test_dir'] = args.test_dir
        hyperparams['change_fps'] = args.change_fps

        while(True):
            try:
                main(hyperparams, args.num_epochs)
                break
            except RuntimeError as error:
                print("////////////////////////////////////////////////")
                print(error)
                print("////////////////////////////////////////////////")
                hyperparams['batchsize'] = int(hyperparams['batchsize']/1.333)
                time.sleep(2)
                continue
    elif args.test == 1:
        print('\n\n------------------------------START HyperBand TESTING------------------------------')
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_dir = "/HB_train_"+args.model_type+"_"+str(today)
        args.test_dir = test_dir
        base_dir = "./Pre/results" + test_dir
        os.mkdir(base_dir)
        hb = Hyperband(args, get_params, main)
        hb.run(skip_last = 1, dry_run = False, hb_result_file = base_dir+"/hb_result.json", hb_best_result_file = base_dir+"/hb_best_result.json" )
        print('\n\n------------------------------FINISH HyperBand TESTING------------------------------')
