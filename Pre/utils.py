"""
Some useful function for learning process
"""
from __future__ import absolute_import

import json
import numpy as np
import random
import torch as th
from torch.utils.data import Dataset
from Pre.constants import LEN_SEQ, SEQ_PER_EPISODE_C,  RANDS, RES_DIR
from PIL import Image


"""
This function prepare results of testing in format which is writable in *.json file
"""
def gen_dict_for_json(keys, values):
    """
    Args:
        keys (np.array): keys
        values (PyTorch Tensor): values

    Return (dict): dictionary with right format for *.json file
    """
    d = {}
    for i in range(len(values)):
        d [str(keys[i])] = list(np.array(values[i].numpy(), dtype=float))
    return d


"""
Normalize pitch and roll angles
"""
def min_max_norm(x, min_e = -90.0, max_e = 90.0):
    """
    Args:
        x (float): angle
        min_e (float): mininal angle
                        Default: -90.0
        max_e (float): maximal angle
                        Default: 90.0

    Return (float) : normalazed value of angle
    """
    return ((x - min_e) * 2) / (max_e - min_e) - 1


"""
Set parameters for pretrained part of models
"""
def use_pretrainted(model, pretrained_part):
    """
    Args:
        model (PyTorch nn.Module): model for setting
        pretrained_part (PyTorch nn.Module): pretrained model
    """

    # 1024 features vector Autoencoder
    pretrained_part.load_state_dict(th.load('./Pre/cnn_autoencoder_model_1s_1im_tmp.pth'))
    model.encoder[0].weight = pretrained_part.encoder[0].weight
    model.encoder[0].bias = pretrained_part.encoder[0].bias
    model.encoder[3].weight = pretrained_part.encoder[3].weight
    model.encoder[3].bias = pretrained_part.encoder[3].bias
    model.encoder[6].weight = pretrained_part.encoder[6].weight
    model.mu.weight = pretrained_part.fc1.weight
    model.mu.bias = pretrained_part.fc1.bias
    model.std.weight = pretrained_part.fc2.weight
    model.std.bias = pretrained_part.fc2.bias

    # Use this code when change AutoEncoder settings
    # 256/400 features vector Autoencoder
    # CNN_part_tmp.load_state_dict(th.load(RES_DIR+'400_features_autoencoder_model_VAEs_1im_tmp.pth'))
    # model.encoder[0].weight = pretrained_part.encoder[0].weight
    # model.encoder[0].bias = pretrained_part.encoder[0].bias
    # model.encoder[1].weight = pretrained_part.encoder[1].weight
    # model.encoder[1].bias = pretrained_part.encoder[1].bias
    # model.encoder[1].running_mean = pretrained_part.encoder[1].running_mean
    # model.encoder[1].running_var = pretrained_part.encoder[1].running_var
    # model.encoder[1].num_batches_tracked = pretrained_part.encoder[1].num_batches_tracked
    # model.encoder[4].weight = pretrained_part.encoder[4].weight
    # model.encoder[4].bias = pretrained_part.encoder[4].bias
    # model.encoder[5].weight = pretrained_part.encoder[5].weight
    # model.encoder[5].bias = pretrained_part.encoder[5].bias
    # model.encoder[5].running_mean = pretrained_part.encoder[5].running_mean
    # model.encoder[5].running_var = pretrained_part.encoder[5].running_var
    # model.encoder[5].num_batches_tracked = pretrained_part.encoder[5].num_batches_tracked
    # model.encoder[8].weight = pretrained_part.encoder[8].weight
    # model.encoder[9].weight = pretrained_part.encoder[9].weight
    # model.encoder[9].bias = pretrained_part.encoder[9].bias
    # model.encoder[9].running_mean = pretrained_part.encoder[9].running_mean
    # model.encoder[9].running_var = pretrained_part.encoder[9].running_var
    # model.encoder[9].num_batches_tracked = pretrained_part.encoder[9].num_batches_tracked
    # model.fc0.weight = pretrained_part.fc0.weight
    # model.fc0.bias = pretrained_part.fc0.bias
    # model.fc00.weight = pretrained_part.fc00.weight
    # model.fc00.bias = pretrained_part.fc00.bias
    # model.mu.weight = pretrained_part.fc1.weight
    # model.mu.bias = pretrained_part.fc1.bias
    # model.std.weight = pretrained_part.fc2.weight
    # model.std.bias = pretrained_part.fc2.bias


"""
Writing experiment results to a file (result.txt)
"""
def write_result(hyperparams, models : list, optimizers : list,
                result_file_name = "/result.txt", best_train_loss = -1,
                best_val_loss = -1, final_test_loss = -1, time = -1, seq_per_ep = 0,
                seq_len = -1, num_epochs = -1):
    """
    Args:
        hyperparams (dict): model's hyperparametres
        models (list): list of models (in case the whole model has several separated parts)
        optimizers (list): list of optimizers (in case the whole model has several separated parts)
        result_file_name (str): the name of the results file
        best_train_loss (float) : best TRAIN loss of experiment
        best_val_loss (float) : best VALIDATION loss of experiment
        final_test_loss (float) : final TEST loss of experiment
        time (float): the duration of the experiment
        seq_per_ep (int): number of sequences in an episode
        seq_len (int): the length of the sequence
        num_epochs(int): number of epochs

    """

    with open(result_file_name, "a") as f:

        f.write("current model:                                              ")
        f.write(hyperparams['model_type'])
        f.write("\nTraining time:   ")
        f.write(str(time))
        f.write("\nbest avg train error [data normalized (-1 : 1) ]     :    ")
        f.write(str(best_train_loss))
        f.write("\nbest avg validation loss [data normalized (-1 : 1) ] :    ")
        f.write(str(best_val_loss))
        f.write("\nfinal avg test loss [data normalized (-1 : 1) ]      :    ")
        f.write(str(final_test_loss))
        f.write("\n")
        f.write("time to predict is (seconds):      ")
        f.write(str(hyperparams['time_to_predict']))
        f.write("\n")
        f.write("time (seconds) used to predict:    ")
        f.write(str(hyperparams['use_sec']))
        f.write("\n")
        f.write("Use n episodes:                    ")
        f.write(str(hyperparams['use_n_episodes']))
        f.write("\n")
        f.write("Seqences per epizode:              ")
        f.write(str(seq_per_ep))
        f.write("\n")
        f.write("Seqence length:                    ")
        f.write(str(seq_len))
        f.write("\n")
        f.write("Frame interval:                    ")
        f.write(str(hyperparams['frame_interval']))
        f.write("\n")
        f.write("Batch_Size:                        ")
        f.write(str(hyperparams['batchsize']))
        f.write("\n")
        f.write("Randomseed:                        ")
        f.write(str(hyperparams['seed']))
        f.write("\n")
        f.write("Cuda:                              ")
        f.write(str(hyperparams['cuda'] ))
        f.write("\n")
        f.write("Learning rate:                     ")
        f.write(str(hyperparams['learning_rate']))
        f.write("\n")
        f.write("Encoder latent vector size:                ")
        f.write(str(hyperparams['encoder_latent_vector']))
        f.write("\n")
        f.write("Decoder latent vector size:                ")
        f.write(str(hyperparams['decoder_latent_vector']))
        f.write("\n")

        f.write("L2 regularisation weight_decay :   ")
        f.write(str(hyperparams["weight_decay"]))

        f.write("\n")
        f.write("Epochs:                            ")
        f.write(str(num_epochs))
        for model in models:
            f.write("\n")
            f.write(str(model))

        for optimizer in optimizers:
            f.write("\n")
            f.write(str(optimizer))

        f.write("\n")
        f.write("Data location:                     ")
        f.write(str(hyperparams['train_folder']))
        f.write("\n")
        f.write("Use loaded weights (all):          ")
        f.write(str(hyperparams['load_weight']))
        f.write("\n")
        f.write("Use loaded weights date:           ")
        f.write(str(hyperparams['load_weight_date']))
        f.write("\n\n--------------------------------------------------------------------------------------------\n\n")
    f.close()


"""
Load index for get right data from DataLoader
"""
def loadLabels(folder_prefix, N_episodes_st, N_episodes_end, seq_per_ep ,p_train=0.7, p_val=0.15, p_test=0.15):
    """
    Args:
        folder_prefix (str):
        N_episodes_st (int): first episode number which is using for experiment
        N_episodes_end (int): last episode number which is using for experiment
        seq_per_ep (int): number of sequences in an episode
        p_train (float) : percentage of training data
        p_val (float) : percentage of validation data
        p_test (float) : percentage of testing data
        return (np.array, np.array, np.array): array of indices of training sequences,
                                                array of indices of validation sequences,
                                                array of indices of test sequences

    """
    random.seed(RANDS)


    all_ep = np.linspace(N_episodes_st*seq_per_ep, N_episodes_end*seq_per_ep, (N_episodes_end*seq_per_ep-N_episodes_st*seq_per_ep +1), dtype =int )

    train_ep = np.array([], dtype =int)
    val_ep = np.array([], dtype =int)
    test_ep = np.array([], dtype =int)

    # TT = int((len(all_ep)*4)/ (N_episodes_end-N_episodes_st))
    # 60 is random number. Setting this number can improve model performance
    tmp_num = 60
    TT = int((len(all_ep))/ tmp_num)
    # for min_ep in range(int((N_episodes_end-N_episodes_st)/4)):
    for min_ep in range(tmp_num):
        tmp_ep = all_ep[min_ep*TT:(min_ep+1)*TT]
        # tmp_ep = np.linspace(N_episodes_st + tmp*min_ep, N_episodes_st + tmp*(min_ep+1), (tmp +1), dtype =int )
        len_tmp_ep = tmp_ep.size
        random.shuffle(tmp_ep)
        train_ep = np.concatenate((train_ep, tmp_ep[0 : int(p_train*len_tmp_ep)] ))
        val_ep = np.concatenate((val_ep, tmp_ep[int(p_train*len_tmp_ep) : int((p_train+p_val)*len_tmp_ep)] ))
        test_ep = np.concatenate((test_ep, tmp_ep[int((p_train+p_val)*len_tmp_ep): ] ))

    # train_ep = np.concatenate((train_ep,all_ep[int((N_episodes_end-N_episodes_st)/4)*TT:]))
    train_ep = np.concatenate((train_ep,all_ep[ tmp_num*TT:]))

    random.shuffle(train_ep)
    random.shuffle(val_ep)
    random.shuffle(test_ep)

    return train_ep, val_ep, test_ep


"""
Universal DataLoader for different models
"""
class JsonDataset_universal (Dataset):
    def __init__(self, labels, folder_prefix="", preprocess=False, predict_n_im = 10, use_n_im = 10, seq_per_ep = 36, use_LSTM = False, use_stack = False, change_fps = False):
        """
        Args:
            labels (int) : sequence indeces
            folder_prefix (str): part of path to the data folder
            preprocess (bool): True if need to preprocess data (normalization)
            predict_n_im (int): number of images should be predicted
            use_n_im (int): number of images used for prediction
            seq_per_ep (int): number of sequences in an episode
            use_LSTM (bool): True if model uses LSTM architecture
            use_stack (bool): True if model uses stack of images like an input
            change_fps (bool): True if want use 1 fps with the data generated with 2 fps
        """
        self.keys = labels.copy()
        self.folder_prefix = folder_prefix
        self.preprocess = preprocess
        self.predict_n_im = predict_n_im
        self.use_n_im = use_n_im
        self.seq_per_ep = seq_per_ep
        self.use_LSTM = use_LSTM
        self.use_stack = use_stack
        self.change_fps = change_fps

    def __getitem__(self, index):
        """
        Args:
            index (int) : index of sequence
        return:
                -models with LSTM-
                image sequence, label sequence
                (PyTorch Tensor, PyTorch Tensor)
                or
                -models without LSTM-
                image sequence, label sequence, the sequence of labels that need to be predicted
                (PyTorch Tensor, PyTorch Tensor, PyTorch Tensor)
        """
        # Through the division of the data into episodes, it is necessary to use labels's index
        index = self.keys[index]
        seq_images = []
        seq_labels = []
        if not self.use_LSTM:
            seq_future = []

        # To find frames position
        episode = index//self.seq_per_ep + 1
        seq_in_episodes = index%self.seq_per_ep + 1

        file_name =  self.folder_prefix  + str(episode) + '/labels_0.json'
        labels_episode = json.load(open(file_name))
        # min_max_stat = json.load(open("Pre/3dmodel/min_max_statistic_320.json"))

        tmp_use_n_im = self.use_n_im
        if self.use_LSTM:
            tmp_use_n_im =  LEN_SEQ

        for i in range(tmp_use_n_im):
            # if want use 1 fps with the data generated with 2 fps
            if self.change_fps:
                if i%2 != 0:
                   continue
            image = (seq_in_episodes-1)*tmp_use_n_im + i
            image_str = str(image) + ".png"

            file = self.folder_prefix + str(episode) + '/' + image_str
            im = Image.open(file).convert("RGB")
            im = np.asarray(im)
            im = im.astype('float')

            tmp_label = np.array(labels_episode[str(image)])

            # normalize it
            if self.preprocess:
                for chanal in range(3):
                    im[...,chanal] = (im[...,chanal]*2)/255 - 1

                tmp_label[0] = min_max_norm(tmp_label[0])
                tmp_label[1] = min_max_norm(tmp_label[1])

            seq_images.append(im)
            seq_labels.append(tmp_label)


        seq_labels = np.array(seq_labels, dtype = np.float32)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.use_stack:
            seq_images = np.dstack(seq_images)
            seq_images = np.array(seq_images, dtype = np.float32)
            seq_images = seq_images.transpose((2, 0, 1)).astype(np.float32)
        else:
            seq_images = np.array(seq_images, dtype = np.float32)
            seq_images = seq_images.transpose((0, 3, 1, 2)).astype(np.float32)



        if not self.use_LSTM:
            for i in range(self.predict_n_im):
                # if want use 1 fps with the data generated with 2 fps
                if self.change_fps:
                    if i%2 != 0:
                       continue
                image = (seq_in_episodes-1)*tmp_use_n_im + tmp_use_n_im + i
                tmp_label_future = np.array(labels_episode[str(image)])

                # normalize it
                if self.preprocess:
                    tmp_label_future[0] = min_max_norm(tmp_label_future[0])
                    tmp_label_future[1] = min_max_norm(tmp_label_future[1])

                seq_future.append(tmp_label_future)

            seq_future = np.array(seq_future, dtype = np.float32)


        if not self.use_LSTM:
            return th.from_numpy(seq_images), th.from_numpy(seq_labels), th.from_numpy(seq_future)
        else:
            return th.from_numpy(seq_images), th.from_numpy(seq_labels)

    def __len__(self):
        return len(self.keys)
