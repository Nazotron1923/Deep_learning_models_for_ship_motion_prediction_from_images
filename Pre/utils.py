from __future__ import print_function, division, absolute_import

import json
import numpy as np
import random
import torch as th
from torch.utils.data import Dataset
from Pre.constants import MAX_WIDTH, MAX_HEIGHT, ROI, INPUT_HEIGHT, INPUT_WIDTH, LEN_SEQ, SEQ_PER_EPISODE_C,  RANDS, RES_DIR
from PIL import Image




def gen_dict_for_json(keys, values):
    d = {}
    for i in range(len(values)):
        d [str(keys[i])] = list(np.array(values[i].numpy(), dtype=float))
    return d


def min_max_norm(x, min_e = -90.0, max_e = 90.0):
    return ((x - min_e) * 2) / (max_e - min_e) - 1


def use_pretrainted(model, pretrained_part):
    CNN_part_tmp = pretrained_part
    CNN_part_tmp.load_state_dict(th.load(RES_DIR+'cnn_autoencoder_model_1s_1im_tmp.pth'))
    model.encoder[0].weight = CNN_part_tmp.encoder[0].weight
    model.encoder[0].bias = CNN_part_tmp.encoder[0].bias
    model.encoder[3].weight = CNN_part_tmp.encoder[3].weight
    model.encoder[3].bias = CNN_part_tmp.encoder[3].bias
    model.encoder[6].weight = CNN_part_tmp.encoder[6].weight
    model.mu.weight = CNN_part_tmp.fc1.weight
    model.mu.bias = CNN_part_tmp.fc1.bias
    model.std.weight = CNN_part_tmp.fc2.weight
    model.std.bias = CNN_part_tmp.fc2.bias


def write_result(hyperparams, models : list, optimizers : list,
                result_file_name = "/result.txt", best_train_loss = -1,
                best_val_loss = -1, final_test_loss = -1, time = -1, seq_per_ep = 0,
                seq_len = -1, num_epochs = -1):
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
        f.write(str(hyperparams['time_gap']))
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
        f.write("Latent vector size:                ")
        f.write(str(hyperparams['latent_vector']))
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


def loadLabels(folder_prefix, N_episodes_st, N_episodes_end, seq_per_ep ,p_train=0.7, p_val=0.15, p_test=0.15):
    random.seed(RANDS)


    all_ep = np.linspace(N_episodes_st*seq_per_ep, N_episodes_end*seq_per_ep, (N_episodes_end*seq_per_ep-N_episodes_st*seq_per_ep +1), dtype =int )

    train_ep = np.array([], dtype =int)
    val_ep = np.array([], dtype =int)
    test_ep = np.array([], dtype =int)

    # TT = int((len(all_ep)*4)/ (N_episodes_end-N_episodes_st))
    TT = int((len(all_ep))/ 60)
    # for min_ep in range(int((N_episodes_end-N_episodes_st)/4)):
    for min_ep in range(60):
        tmp_ep = all_ep[min_ep*TT:(min_ep+1)*TT]
        # tmp_ep = np.linspace(N_episodes_st + tmp*min_ep, N_episodes_st + tmp*(min_ep+1), (tmp +1), dtype =int )
        len_tmp_ep = tmp_ep.size
        random.shuffle(tmp_ep)
        train_ep = np.concatenate((train_ep, tmp_ep[0 : int(p_train*len_tmp_ep)] ))
        val_ep = np.concatenate((val_ep, tmp_ep[int(p_train*len_tmp_ep) : int((p_train+p_val)*len_tmp_ep)] ))
        test_ep = np.concatenate((test_ep, tmp_ep[int((p_train+p_val)*len_tmp_ep): ] ))

    # train_ep = np.concatenate((train_ep,all_ep[int((N_episodes_end-N_episodes_st)/4)*TT:]))
    train_ep = np.concatenate((train_ep,all_ep[60*TT:]))

    random.shuffle(train_ep)
    random.shuffle(val_ep)
    random.shuffle(test_ep)

    return train_ep, val_ep, test_ep


def loadTestLabels (folder_prefix, pred_time, N_episodes_st, N_episodes_end):
    random.seed(RANDS)
    all_ep = np.linspace(N_episodes_st, N_episodes_end, (N_episodes_end-N_episodes_st +1), dtype =int )
    len_all_ep = all_ep.size

    random.shuffle(all_ep)

    test_ep = all_ep[:]
    print('test_ep-> ', test_ep)

    test_labels = {}

    for episode in test_ep:
        file_name_episode =  folder_prefix  + str(episode) + '/labels_' + str(pred_time) + '.json'
        labels_episode = json.load(open(file_name_episode))
        images_edisode = np.array( list(map(int, list(labels_episode.keys()) )) )
        images_edisode.sort()
        N_images_in_episode = images_edisode.size
        images_edisode += N_images_in_episode*(episode-1)   # to index every label episode

        #images = np.concatenate([images, images_edisode])
        for l in range(N_images_in_episode*(episode-1), N_images_in_episode*episode):
            test_labels[l] = labels_episode[str(l-N_images_in_episode*(episode-1))]

    return test_labels


"""
description to add
"""
class JsonDataset_universal (Dataset):
    def __init__(self, labels, folder_prefix="", preprocess=False, predict_n_im = 10, use_n_im = 10, seq_per_ep = 36, use_LSTM = False, use_stack = False):
        self.keys = labels.copy()
        self.folder_prefix = folder_prefix
        self.preprocess = preprocess
        self.predict_n_im = predict_n_im
        self.use_n_im = use_n_im
        self.seq_per_ep = seq_per_ep
        self.use_LSTM = use_LSTM
        self.use_stack = use_stack

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (PyTorch Tensor, PyTorch Tensor)
        """
        # Through the division of the data into episodes, it is necessary to use labels's index
        index = self.keys[index]
        seq_images = []
        seq_labels = []
        if not self.use_LSTM:
            seq_future = []

        # To find photo position
        episode = index//self.seq_per_ep + 1
        seq_in_episodes = index%self.seq_per_ep + 1

        file_name =  self.folder_prefix  + str(episode) + '/labels_0.json'
        labels_episode = json.load(open(file_name))
        # min_max_stat = json.load(open("Pre/3dmodel/min_max_statistic_320.json"))

        tmp_use_n_im = self.use_n_im
        if self.use_LSTM:
            tmp_use_n_im =  LEN_SEQ

        for i in range(tmp_use_n_im):
            image = (seq_in_episodes-1)*tmp_use_n_im + i
            # maybe need to inverse (image - self.N_im + i)
            image_str = str(image) + ".png"

            file = self.folder_prefix + str(episode) + '/' + image_str
            im = Image.open(file).convert("RGB") # RGB(270, 486, 3)
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
