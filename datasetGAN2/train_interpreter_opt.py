"""
This code was generated based on 'DatasetGAN: Efficient Labeled Data Factory
with Minimal Human Effort' publication, available at:
https://github.com/nv-tlabs/datasetGAN_release/

Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
import numpy as np
import imageio
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from collections import OrderedDict
import gc
import pickle
import scipy.stats
import argparse
import cv2
import time
import psutil
from models.stylegan1 import G_mapping,Truncation,G_synthesis
from utils.datasetgan_utils import multi_acc, colorize_mask, latent_to_image, oht_to_scalar, Interpolate
from utils.utils import memory_usage

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim, dropout_prob_1=0.6, dropout_prob_2=0.4):
        super(pixel_classifier, self).__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(dropout_prob_1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Dropout(dropout_prob_2),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(dropout_prob_1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(dropout_prob_2),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)

def save_log(data, file="log.txt", print_data=True):
    """Function print and store the logs."""
    if(print_data):
        print(str(data))
    with open(os.path.join(file), 'a') as f:
        f.write(data)
        f.write("\n")

def get_chunk_from_memmap(X, y, chunk_index, chunk_size):
    """Function to load chunks of the feature maps. Its inputs are the features
    (X) and masks (y) as memory maps, the chunk index, and the chunk size (number
     of pixels). It returns a tuple containing the chunk of features and chunk of
     masks"""
    save_log(f"Getting chunk from X and y memmaps. Chunk index: {chunk_index}, chunk size: {chunk_size} pixels")
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size
    return X[start_index:end_index], y[start_index:end_index]

def load_X_y_as_mem_map(name_x, name_y):
    """Function to load the feature maps as memory maps. It avoids loading all
    the featue maps into the RAM at the same time. It resturns the features (X) and
    masks (y) as memory maps."""
    #This code was done based on https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/00_How_to_efficiently_work_with_very_large_numpy_arrays.ipynb
    start = time.time()
    save_log("Loading X data...")
    X = np.load(name_x, mmap_mode='c')
    save_log(f"Time spent to load X with shape {X.shape}: {time.time() - start}")
    start = time.time()
    save_log("Loading y data...")
    y = np.load(name_y, mmap_mode='c')
    save_log(f"Time spent to load y with shape {y.shape}: {time.time() - start}")
    gc.collect()
    save_log(f"Available memory after reading X and y as items: {memory_usage(detailed_output=True)}")
    return X, y

def prepare_stylegan(args):
    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7
        else:
            assert "Not implementated!"

        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()

    else:
        assert "Not implementated error"

    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers


def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True, best_epoch=10):
    """Function to generate automatically anotated samples. This function is
    adapted from DatasetGAN original code with a few modifications. Its inputs
    are the parsed arguments for the execution (args), the checkpoint path
    (checkpoint_path), the number of image samples (num_sample), the index of
    thefirst image to be generated (start_step), the flag to provide visualization
    (vis), and the index of the epoch from which the checkpoint is loaded
    (best_epoch).
    """
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    else:
        assert False

    save_log(f"Generating automatically annotated images for {args['category']} dataset...")
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples' )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        save_log(f'Experiment folder created at: {result_path}')


    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        save_log(f'MODEL_NUMBER: {MODEL_NUMBER}')

        classifier = pixel_classifier(numpy_class=args['number_class'],
                                      dim=args['dim'][-1],
                                      dropout_prob_1=args['dropout_prob_1'],
                                      dropout_prob_2=args['dropout_prob_2']
                                      )
        classifier =  nn.DataParallel(classifier, device_ids=device_ids).cuda()
        #In case you want to use the checkpoint from another epoch, modify the variable best_epoch
        checkpoint = torch.load(os.path.join(checkpoint_path,
                                  'model_' + str(MODEL_NUMBER) +'_epoch_' + str(best_epoch) +'.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        classifier_list.append(classifier)

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step

        save_log(f"Total number of sample images: {num_sample}")

        for i in range(num_sample):
            if i % 100 == 0:
                save_log(f"Generating {i} out of {num_sample} sample images...")
            curr_result = {}
            latent = np.random.randn(1, 512)
            curr_result['latent'] = latent
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)
            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]
            image_cache.append(img)

            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]
            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None
            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)
                img_seg = img_seg.squeeze()

                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()
                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)
            full_entropy = Categorical(mean_seg).entropy()
            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img
                #scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '.png'), color_mask.astype(np.uint8))
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '.png'), color_mask.astype(np.uint8))
                #scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '_image.png'), img.astype(np.uint8))
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '_image.png'), img.astype(np.uint8))
            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'label_' + str(count_step) + '.png')
                image_name = os.path.join(result_path,  str(count_step) + '.png')
                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = Image.fromarray(img)
                img_seg = Image.fromarray(img_seg_final.astype('uint8'))
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                img.save(image_name)
                img_seg.save(image_label_name)
                np.save(js_name, js)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1

                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)


def prepare_data(args, palette):
    """Function to prepare the feature maps (equivalent as the one made available
    by the original DatasetGAN paper). It resturns the features
    (all_feature_maps_train) and masks (all_mask_train) of tthe feature maps,
     and the number of images (num_data)."""
    save_log(f"Preparing input data, it might take a while...")
    g_all, avg_latent, upsamplers = prepare_stylegan(args)
    latent_all = np.load(args['annotation_image_latent_path'])
    latent_all = torch.from_numpy(latent_all).cuda()

    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in range(len(latent_all)):

        if i >= args['max_training']:
            break

        name = 'image_mask%0d.npy' % i
        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        mask =  cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.png' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)


    # 3. Generate ALL training data for training pixel classifier
    all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)
    all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)

    vis = []
    for i in range(len(latent_all) ):
        save_log(f"Creating the feature map for image {i+1}")
        gc.collect()

        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1], return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'])
        if args['dim'][0]  != args['dim'][1]:
            # only for car
            #here for car_20 experiment img has shape (1, 512, 512, 3) and feature_maps has shape torch.Size([1, 5056, 512, 512])
                img = img[:, 64:448]
                feature_maps = feature_maps[:, :, 64:448]
            #then for car_20 experiment img shape become (1, 384, 512, 3) and feature_maps shape become torch.Size([1, 5056, 384, 512])
        mask = all_mask[i:i + 1]
        new_mask =  np.squeeze(mask)
        mask = mask.reshape(-1)
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        feature_maps = feature_maps.reshape(-1, args['dim'][2])

        all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)

        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )
        vis.append( curr_vis )

    gc.collect()
    vis = np.concatenate(vis, 1)
    #scipy.misc.imsave(os.path.join(args['exp_dir'], "train_data.png"), vis)
    imageio.imwrite(os.path.join(args['exp_dir'], "train_data.png"), vis)

    #In case we want to use the neighbour pixls as inputs

    return all_feature_maps_train, all_mask_train, num_data


def main(args):
    #Start of execution here
    startt = time.time()
    width = args['dim'][1]
    height = args['dim'][0]

    save_log(f'Entered main') #for debugging purposes

    #Choice of the color pallete according to the experiment
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette

    #Name of the feature maps stored on the disk
    #If you want to store elsewhere, change the varibale to the full pathname
    #Example: ame_x = "my_path_folder/X_" + args['category'] + ".npy"
    name_x = "X_" + args['category'] + ".npy"
    name_y = "y_" + args['category'] + ".npy"
    save_log(f"Loading X and y from local path: {name_x}, {name_y}")

    #The code below avoids preparing the feature maps multiple times
    if(not os.path.isfile(name_x)):
        save_log("Preparing data...")
        start = time.time()
        all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)
        save_log(f'Time spent preparing data: {time.time() - start}')
        save_log("Saving features...")
        start = time.time()
        np.save(name_x, all_feature_maps_train_all)
        save_log(f'Time spent saving features: {time.time() - start}')
        save_log("Saving labels...")
        start = time.time()
        np.save(name_y, all_mask_train_all)
        save_log(f'Time spent saving labels: {time.time() - start}')
        save_log("")
        gc.collect()

    #In case you only want to generate the feature maps, add the following line
    #to the json file in experiments folder: "only_gen_data_files": true
    if "only_gen_data_files" in args:
        if(args['only_gen_data_files']):
            sys.exit()

    start = time.time()
    X_mem_map, y_mem_map = load_X_y_as_mem_map(name_x, name_y)
    save_log(f"X shape: {X_mem_map.shape}")
    save_log(f"y shape: {y_mem_map.shape}")
    num_data = int(y_mem_map.shape[0]/(width)/(height)) #Number of images
    max_label = args['max_label'] #Index of the last lable
    save_log(f" ************************* max_label {str(max_label)} **************************")
    save_log(f"*********** Current number of images used in training {str(int(num_data/4)*4)} ************") # it makes sure it is a multiple of 4
    batch_size = args['batch_size']
    total_epochs = 10
    #The training is executed
    for MODEL_NUMBER in range(args['model_num']): #for each network in the ensemble...
        stop_sign = False
        gc.collect()
        save_log(f'Selecting model {MODEL_NUMBER}, time spent so far: {time.time() - startt}')
        start = time.time()

        #Defines the classifier
        classifier = pixel_classifier(numpy_class=args['number_class'],
                                      dim=args['dim'][-1],
                                      dropout_prob_1=args['dropout_prob_1'],
                                      dropout_prob_2=args['dropout_prob_2']
                                      )
        #Initial setup of the training
        classifier.init_weights()
        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        for epoch in range(100):
            #The mean calculations are started again at each epoch
            mean_loss = 0
            mean_acc = 0
            iteration = 0
            split_subset_train = 8
            best_loss = 10000000
            start_1000_it = time.time() #Variable to record the time spent in 1000 iterations
            #The training is performed in chunks
            for chunk in range(num_data):
                start = time.time()
                save_log(f"Memory usage before reading chunk {chunk}: {memory_usage()}")
                #Getting the memory maps of each chunk
                X_local, y_local  = get_chunk_from_memmap(X_mem_map, y_mem_map, chunk, (width)*(height))
                #Then, each chunk is shuffled
                random_indexes=torch.randperm(y_local.shape[0])
                #A training dataset is generated
                train_data = trainData(torch.FloatTensor(X_local)[random_indexes],
                                       torch.FloatTensor(y_local)[random_indexes])
                del random_indexes
                gc.collect()
                save_log(f"Memory usage after generating train_data for chunk {chunk}: {memory_usage()}")
                save_log(f'Time spent splitting and preparing the train data: {time.time() - start}')
                #Splits the training in subsets for reduced memory consumption
                end_position = 0
                #The training is performed in chunks, where each chunk is subdivided in dataloaders
                for subset in range(split_subset_train):
                    start = time.time()
                    start_position = int(end_position) #Start position of the subset
                    #End position of the subset
                    if (subset == split_subset_train - 1): end_position = len(train_data)
                    else: end_position += int(len(train_data)/split_subset_train) #To make sure the int() casting will not make end_position end before the end
                    train_subset = Subset(train_data, list(range(start_position, end_position)))
                    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,  num_workers=len(device_ids))
                    gc.collect()

                    #Training loop
                    classifier.train()
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_batch = y_batch.type(torch.long)
                        y_batch = y_batch.type(torch.long)
                        optimizer.zero_grad()
                        y_pred = classifier(X_batch)
                        loss = criterion(y_pred, y_batch)
                        acc = multi_acc(y_pred, y_batch)
                        mean_loss += loss.item()
                        mean_acc += acc.item()
                        loss.backward()
                        optimizer.step()
                        iteration += 1
                        if iteration % 10000 == 0:
                            save_log(f'Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, Chunk: {chunk}, Subset {subset}, iteration: {iteration}, train loss: {loss.item()}, train acc: {acc.item()}, mean train loss: {mean_loss/iteration}, mean train acc: {mean_acc/iteration}')
                            save_log(f'Time spent in 10000 iterations: {time.time() - start_1000_it}')
                            start_1000_it = time.time()

                        if iteration % 5000 == 0:
                            model_path = os.path.join(args['exp_dir'],
                                                      'model_20parts_iter' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                            save_log(f'Save checkpoint, Epoch : {str(epoch)},  Path: {str(model_path)}')
                            torch.save({'model_state_dict': classifier.state_dict()},
                                       model_path)
                        if epoch >= total_epochs:
                            if loss.item() < best_loss:
                                best_loss = loss.item()
                                break_count = 0
                            else:
                                break_count += 1

                            if break_count > 50:
                                stop_sign = True
                                save_log(f"*************** Break, Total iters, {iteration} at epoch {str(epoch)} ***************")
                                break

                    save_log(f'Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, mean loss: {mean_loss/iteration}, mean accuracy: {mean_acc/iteration}')
                    save_log(f'Time spent training model {MODEL_NUMBER}, chunk {chunk} subset {subset}: {time.time() - start}')
                    #Exits Subset for loop
                    if stop_sign:
                        break
                #Free memory before loading new chunks
                del train_subset
                del train_loader
                del train_data
                gc.collect()
                #Exits Chunk for loop
                if stop_sign:
                    break
            #Mean train loss and accuracy are calculated for each epoch and each model
            mean_loss = mean_loss/iteration
            mean_acc = mean_acc/iteration
            gc.collect()
            save_log(f'Mean results at the end of epoch {epoch}: mean training loss = {mean_loss}, mean training accuracy = {mean_acc}')
            #Saving the model at the end of each epoch
            start = time.time()
            model_path = os.path.join(args['exp_dir'],
                                      'model_' + str(MODEL_NUMBER) +'_epoch_'+ str(epoch) +'.pth')
            torch.save({'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_path)
            save_log(f'Time spent saving checkpoint of model {MODEL_NUMBER} and epoch {epoch}: {time.time() - start}')
            #Exits Epoch for loop
            if stop_sign:
                break
    #Finishing the execution
    gc.collect()
    torch.cuda.empty_cache()    # clear cache memory on GPU
    save_log(f'Time spent in total: {time.time() - startt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--classifier_param', type=str, default = "experiments/nn_classifier.json")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)
    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    opts2 = json.load(open(args.classifier_param, 'r'))
    print("Clf parameters", opts2)
    opts = {**opts, **opts2}

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    os.system('cp %s %s' % (args.classifier_param, opts['exp_dir']))

    if args.generate_data: #In case we want to generate the images of DatasetGAN
        device_ids = [0]
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else: #In case we want to train the pixel_classifier
        device_ids = [i for i in range(args.n_processes)] #If we want to use multiple GPUs a list of them is generated
        main(opts)
