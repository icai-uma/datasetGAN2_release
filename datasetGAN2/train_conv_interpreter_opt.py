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
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Categorical
import torch.optim as optim
import scipy.stats
import json
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import argparse
import cv2
import h5py
import pickle
import random
import math
import gc
import pandas as pd
from collections import OrderedDict
from utils.datasetgan_utils import colorize_mask, latent_to_image, oht_to_scalar, Interpolate
from utils import Adam16, LogSaver, join_img_maks, set_seed, memory_usage, put_color_mask
from models.stylegan1 import G_mapping, Truncation, G_synthesis

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.empty_cache()

class convolutional_pixel_classifier(nn.Module):
    def __init__(self, numpy_class, depth, kernels=5, kernel_size=3, dropout_prob_1=0.6, dropout_prob_2=0.4):
        super(convolutional_pixel_classifier, self).__init__()
        self.conv1 = nn.Conv2d(depth, int(depth*kernels), kernel_size)
        self.linear1 = nn.Linear(kernels*depth, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.drop1 = nn.Dropout(dropout_prob_1)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.drop2 = nn.Dropout(dropout_prob_2)
        self.linear3 = nn.Linear(128, numpy_class)
        
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
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        return x

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

class CustomDataset(Dataset):
    def __init__(self, X_mem_map, y_mem_map, padding=1, kernel_size=3, crossval=False, folds=4, dtype=torch.float16, conv=True, verbose=False, logger=None):
        self.X_mem_map = X_mem_map
        self.y_mem_map = y_mem_map
        self.indexes = None
        self.batch_size = None
        self.num_batches = None
        self.defined_batch_size = False
        self.padding = padding
        self.kernel_size = kernel_size
        self.crossval = crossval
        self.folds = folds
        self.initial_available_indexes = None
        self.dtype = dtype
        self.conv = conv
        self.output_xval = None
        self.log = logger
        if verbose:
            self.display_inputs()
    
    def __getitem__(self, index):
        idx = [self.initial_available_indexes[i][index] for i in range(len(self.initial_available_indexes))]   	
        return self.X_mem_map[tuple(idx)], self.y_mem_map[tuple(idx)]
    def __len__(self):
        return len(self.X_mem_map)
    
    def display_inputs(self):
        if self.log:
            self.log.save_log("******************** Custoom dataset config ***************************")
            self.log.save_log(f"X_mem_map.shape={self.X_mem_map.shape}, y_mem_map.shape={self.y_mem_map.shape}, conv={self.conv}")
            self.log.save_log(f"padding={self.padding}, kernel_size={self.kernel_size}, crossval={self.crossval}, folds={self.folds}, dtype={self.dtype}")
            self.log.save_log("************************************************************************")
        else:
            print(("******************** Custoom dataset config ***************************"))
            print(f"X_mem_map.shape={self.X_mem_map.shape}, y_mem_map.shape={self.y_mem_map.shape}, conv={self.conv}")
            print(f"padding={self.padding}, kernel_size={self.kernel_size}, crossval={self.crossval}, folds={self.folds}, dtype={self.dtype}")
            print("************************************************************************")
    
    def get_output_xval(self):
        if self.output_xval is None:
            raise "ERROR: the method start_loop(val_loop=True) must be called before calling get_output_xval()!"
        return self.output_xval

    def start_loop(self, batch_size=16, fold=None, val_loop=False):
        # Set values possible to be sampled except in the border of each image's feature map (padding). False means available to be sampled
        self.indexes = np.ones(self.y_mem_map.shape, dtype="bool")
        
        if self.padding == 0:
            limit_inf, limit_sup = None, None
        elif self.padding == "valid":
            limit_inf, limit_sup =-int(self.kernel_size/2), int(self.kernel_size/2) # be aware of the negative sign of limit_inf!
        else:
            raise "padding method not implemented!"
        # If we want to crossvalidate instead just training
        if self.crossval:
            if fold is not None:
                fold_images = np.array_split(np.arange(self.indexes.shape[0]), self.folds)
                # If we want to prepare the validation loop
                if val_loop:
                    self.output_xval = np.zeros((len(fold_images[fold]), self.y_mem_map.shape[1]-limit_sup+limit_inf, self.y_mem_map.shape[2]-limit_sup+limit_inf) , dtype="bool")
                    fold_images = list(fold_images[fold])
                    for img in fold_images:
                        self.indexes[img:img+1, limit_sup:limit_inf, limit_sup:limit_inf] = False
                        # Uncomment the following line below and comment the line above if you want to run it quickly for test only
                        #self.indexes[1:2, 10:18, 10:18] = False
                # If we want to prepare the training loop
                else:
                    fold_images.pop(fold)
                    for f in fold_images:
                        for img in f:
                            self.indexes[img:img+1, limit_sup:limit_inf, limit_sup:limit_inf] = False
                            # Uncomment the following line below and comment the line above if you want to run it quickly for test only
                            #self.indexes[2:3, 10:18, 10:18] = False
            else:
                assert("The fold index must be passed!")
        # If normal training is expected
        else:
            self.indexes[:, limit_sup:limit_inf, limit_sup:limit_inf] = False
            # Uncomment the following line below and comment the line above if you want to run it quickly for test only
            #self.indexes[1:2, 10:22, 10:22] = False

        # Set the neighbourhood of a feature array for convolution
        self.initial_available_indexes = np.where(self.indexes==False)
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.initial_available_indexes[0])/batch_size)
        
    def get_batch_indexes(self):
        return range(self.num_batches)
      
    def get_batch(self, positions):
        X = torch.stack([torch.tensor(self.X_mem_map[positions[0][p], 
                                        positions[1][p]-int(self.kernel_size/2):positions[1][p]+int(self.kernel_size/2)+1, 
                                        positions[2][p]-int(self.kernel_size/2):positions[2][p]+int(self.kernel_size/2)+1,
                                        :]) for p in range(len(positions[0]))]).to(device).reshape(len(positions[0]), max(self.kernel_size,1), max(self.kernel_size,1), -1).permute(0,3,1,2).to(self.dtype)
        if not self.conv:
            X = X.squeeze(2,3)

        y  = torch.stack([torch.tensor(self.y_mem_map[positions[0][p], 
                                        positions[1][p]:positions[1][p]+1, 
                                        positions[2][p]:positions[2][p]+1,
                                        ]) for p in range(len(positions[0]))]).to(device).reshape(len(positions[0]),).type(torch.long)
        return X, y

    def get_positions(self, shuffle=True):
        # Get the indexes of the feature map not trained in the current epoch yet
        available = np.where(self.indexes==False)
        if shuffle:
            # Sample self.batch_size indexes
            if len(available[0])<self.batch_size:
                positions = random.sample(range(len(available[0])), len(available[0]))
            else: 
                positions = random.sample(range(len(available[0])), self.batch_size)
        else:
            positions = list(range(len(available[0])))[:self.batch_size]
        # Mark these positions so that it will not be sampled again
        idx = [available[i][positions] for i in range(len(available))]
        self.indexes[tuple(idx)] = True
        return idx
    
    def load_batch(self, shuffle=True, return_positions=False):
        positions = self.get_positions(shuffle)
        result = self.get_batch(positions)
        if return_positions:
            return result, positions
        return result                  

def prepare_model(args):
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

    return g_all, upsamplers, avg_latent

def prepare_data(args, palette):
    log.save_log(f"Preparing input data, it might take a while...")
    g_all, upsamplers, avg_latent = prepare_model(args)
    
    latent_all = np.load(args['annotation_image_latent_path'])

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
        if "image_format" in args:
            im_name = os.path.join( args['annotation_mask_path'], f"image_{i}.{args['image_format']}")
        else:
            im_name = os.path.join( args['annotation_mask_path'], 'image_%d.png' % i)
        img = Image.open(im_name)
        log.save_log(f"image_{i}.{args['image_format']} shape: {np.array(img).shape}")
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(int(args['number_class'])):
            if (mask_list[i] == target).sum() < 10:
                mask_list[i][mask_list[i] == target] = 0

    all_mask = np.stack(mask_list)


    # 3. Generate ALL training data for training pixel classifier
    # Original approach
    # 1024 (w) * 1024 (h) * 16 (len(latent_all)) * 5088 (features) * 2 (np.float16, 2 bits) = 170.7 GB!
    # all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)
    # all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)

    # Our approach, using a persistent shuffled feature map.
    # It is beneficial because:
    # 1 - It avoids creating a feature map every time the code is executed, helping testing variations of hyperparameters.
    # 2 - It is possible to access the feature maps after the execution is completed and unshuffle it.
    # 3 - It avoids extra burden during the training for shuffling the data.
    # 4 - It allows to load subsets of the previously shuffled data, avoiding the need af a prohibitive amount of RAM.
    if not os.path.isdir(os.path.join(args["local_dir"], "feature_maps")):
        os.mkdir(os.path.join(args["local_dir"], "feature_maps"))
    if "variation" in args:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + args["variation"] + ".hdf5")
    else:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + ".hdf5")
    log.save_log(f"Feature maps filename: {feature_maps_filename}")
    hdf5_feature_maps = h5py.File(feature_maps_filename, "w")
    all_feature_maps_train = hdf5_feature_maps.create_dataset("all_feature_maps_train", (len(latent_all), args['dim'][0], args['dim'][1], args['dim'][2]), dtype=np.float16, chunks=(1, 1, 1, args['dim'][2]), compression="gzip")
    hdf5_feature_maps["/all_feature_maps_train"].attrs["num_data"] = num_data
    all_mask_train = hdf5_feature_maps.create_dataset("all_mask_train", (len(latent_all), args['dim'][0], args['dim'][1]), dtype=np.float16, compression="gzip")

    vis = []
    for i in range(latent_all.shape[0]):
        log.save_log(f"Creating the feature map for image {i+1}")
        gc.collect()
        torch.cuda.empty_cache()
        latent_input = latent_all[i:i+1,:]
        latent_input = torch.from_numpy(latent_input).to(device)
        latent_input = latent_input.squeeze().float()
        img, feature_maps = latent_to_image(g_all, 
                                            upsamplers, 
                                            latent_input.unsqueeze(0), 
                                            dim=args['dim'][1], 
                                            return_upsampled_layers=True, 
                                            use_style_latents=args['annotation_data_from_w']
                                            )
        if args['dim'][0]  != args['dim'][1]:
            # only for car
            # here for car_20 experiment img has shape (1, 512, 512, 3) and feature_maps has shape torch.Size([1, 5056, 512, 512])
            img = img[:, 64:448]
            feature_maps = feature_maps[:, :, 64:448]
            # then for car_20 experiment img shape become (1, 384, 512, 3) and feature_maps shape become torch.Size([1, 5056, 384, 512])
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        new_mask =  np.squeeze(mask)
        mask = np.expand_dims(new_mask,0)

        # Original approach
        # all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        # all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)

        # Our approach, using a persistent shuffled feature map.
        mask = mask.astype(np.float16)
        torch.cuda.empty_cache()
        gc.collect()
        # The feature maps for each image was subdivided to be able to work with low memory GPUs
        chunk_size = 32
        number_chunks_x = int(feature_maps.shape[1]/chunk_size)
        number_chunks_y = int(feature_maps.shape[2]/chunk_size)
        for xc in range(number_chunks_x):
            for yc in range(number_chunks_y):
                feature_maps_temp = feature_maps[:, xc*chunk_size:(xc+1)*chunk_size, yc*chunk_size:(yc+1)*chunk_size, :].detach().cpu().numpy()
                feature_maps_temp = feature_maps_temp.astype(np.float16)
                all_feature_maps_train[i:i+1, xc*chunk_size:(xc+1)*chunk_size, yc*chunk_size:(yc+1)*chunk_size, :] = feature_maps_temp
        del feature_maps, feature_maps_temp
        torch.cuda.empty_cache()
        gc.collect()
        all_mask_train[i:i+1, :, :] = mask
        
        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        if len(im_list[i].shape) == 2:
            curr_vis = np.concatenate([np.stack((im_list[i],)*3, axis=-1), join_img_maks(im_list[i], new_mask, alpha=0.25)], 0)
        else:
            curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )
        vis.append( curr_vis )
        imageio.imsave(os.path.join(args['exp_dir'], "train_data.jpg"),
                      np.concatenate(vis, 1))

    gc.collect()
    vis = np.concatenate(vis, 1)
    imageio.imsave(os.path.join(args['exp_dir'], "train_data.jpg"),
                      vis)

    hdf5_feature_maps.close()

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
        from utils.data_util import crack_palette as palette

    log.save_log(f"Generating automatically annotated images for {args['category']} dataset...")
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples' )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        log.save_log(f'Experiment folder created at: {result_path}')


    g_all, upsamplers, avg_latent = prepare_model(args, seed=10)

    kernel_size = args['kernel_size']
    kernels = args['kernels']
    padding = args['padding']

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        log.save_log(f'MODEL_NUMBER: {MODEL_NUMBER}')
        classes = args['number_class']
        classifier = convolutional_pixel_classifier(classes, args['dim'][-1], kernels=kernels, kernel_size=kernel_size)
        if len(device_ids)>1:
            classifier =  nn.DataParallel(classifier, device_ids=device_ids)
        else:
            classifier = classifier.to(device)
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
        uncertanty_scores = {}
        log.save_log(f"Total number of sample images: {num_sample}")

        for i in range(num_sample):
            if i % 100 == 0:
                log.save_log(f"Generating {i} out of {num_sample} sample images...")
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
                if padding=="valid":
                    img_seg_final = img_seg_final.reshape(args['dim'][0]-int(kernel_size/2), args['dim'][1]-int(kernel_size/2), 1)
                elif padding==0:
                    img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                else:
                    raise "padding method not implemented!"
                img_seg_final = F.pad(img_seg_final, pad=(0, 0, 1, 1, 1, 1), mode='constant', value=0)
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
                js_name = os.path.join(result_path, str(count_step) + '.npy')
                image_mask_name = os.path.join(result_path, 'image_and_mask_' + str(count_step) + '.png')
                uncertanty_scores[i] = curr_result['uncertrainty_score']
                if len(img.shape) == 2:
                    image_name = os.path.join(result_path,  str(count_step) + '.png')
                    img = Image.fromarray(img)
                    img.save(image_name)
                else:
                    Image.fromarray(join_img_maks(img.squeeze(), img_seg_final, alpha=0.25)).save(image_mask_name)
                    image_name = os.path.join(result_path,  str(count_step) + '.tiff')
                    img = Image.fromarray(img.squeeze())
                    img.save(image_name, format="TIFF", save_all=True)
                img_seg_final = img_seg_final.astype('uint8')    
                img_seg = Image.fromarray(img_seg_final)
                np.save(image_label_name.replace('.png','.npy'), img_seg_final)
                img_seg.save(image_label_name)   
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                np.save(js_name, js)
                #count_good += 1
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1
                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    log.save_log(f"Generated {i} samples")
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)
                        
        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '_only_uncertainty_scores.pickle'), 'wb') as f:
            pickle.dump(results, f)

def cross_val(args, fold_run, seed, dtype=torch.float16):
    #Start of execution here
    startt = time.time()

    log.save_log(f'Entered main') #for debugging purposes

    #Choice of the color pallete according to the experiment
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    else:
        from utils.data_util import crack_palette as palette

    #Name of the feature maps stored on the disk
    #If you want to store elsewhere, change the varibale to the full pathname
    #Example: feature_maps_filename = "my_path_folder/X_" + args['category'] + ".hdf5"
    if "variation" in args:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + args["variation"] + ".hdf5")
    else:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + ".hdf5")

    #The code below avoids preparing the feature maps multiple times
    if(not os.path.isfile(feature_maps_filename)):
        log.save_log("Preparing data...")
        start = time.time()
        prepare_data(args, palette)
        log.save_log(f'Time spent preparing data: {time.time() - start}')
    log.save_log(f"Loading training feature maps and its respective masks from {feature_maps_filename}")
    hdf5_feature_maps = h5py.File(feature_maps_filename, "r")
    num_data = hdf5_feature_maps["/all_feature_maps_train"].attrs["num_data"]

    #In case you only want to generate the feature maps, add the following line
    #to the json file in experiments folder: "only_gen_data_files": true
    if "only_gen_data_files" in args:
        if(args['only_gen_data_files']):
            sys.exit()

    start = time.time()
    X_mem_map, y_mem_map = hdf5_feature_maps["/all_feature_maps_train"], hdf5_feature_maps["/all_mask_train"]
    log.save_log(f"Training feature maps shape: {X_mem_map.shape}")
    log.save_log(f"Training masks shape: {y_mem_map.shape}")
    classes = args['number_class']
    log.save_log(f"************************* Number of classes: {classes} **************************")
    log.save_log(f"********************* Number of annotated images: {num_data} ********************")
    batch_size = args['batch_size']
    total_epochs = args['xval_epochs']
    kernel_size = args['kernel_size']
    kernels = args['kernels']
    padding = args['padding']
    folds = args['folds']
    if fold_run >= 0: # If fold_run is positive it means only one fold will be executed
        fold_range = range(fold_run, fold_run + 1)
    else:
        fold_range = range(folds)
    classifier_type = args['classifier_type']

    df = pd.DataFrame(columns=["mode", "fold", "model", "epoch", "batch", "loss", "accuracy", "mean_loss", "mean_accuracy"])
    row = 0
    print_rate_1 = 100
    print_rate_2 = 5
    save_rate = int(num_data*args['dim'][0]*args['dim'][1]/batch_size/3*(folds-1)/(folds))
    log.save_log(f"save_rate: 1/{save_rate}")

    for fold in fold_range:
        for MODEL_NUMBER in range(args['model_num_per_fold_xval']):
            gc.collect()
            log.save_log(f'Selecting model {MODEL_NUMBER} of fold {fold}, time spent so far: {time.time() - startt}')
            start = time.time()
            #Defines the classifier
            classes = args['number_class']
            if classifier_type == "MLP":
                classifier = pixel_classifier(numpy_class=(classes), dim=args['dim'][-1])
                custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size, crossval=True, folds=folds, dtype=dtype, conv=False)
            elif classifier_type == "convolutional-MLP":
                classifier = convolutional_pixel_classifier(classes, args['dim'][-1], kernels=kernels, kernel_size=kernel_size)
                custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size, crossval=True, folds=folds, dtype=dtype)
            else:
                assert f"classifier_type {classifier_type} not implemented!"
            #Initial setup of the training
            classifier.init_weights()
            classifier = nn.DataParallel(classifier, device_ids=device_ids).to(device)
            criterion = nn.CrossEntropyLoss()
            if dtype == torch.float16:
                classifier.half()
                optimizer = Adam16(classifier.parameters(), lr=0.001, device=device)
            else:
                optimizer = optim.Adam(classifier.parameters(), lr=0.001)
            
            for epoch in range(total_epochs):
                log.save_log(f'********** Starting Training loop of fold, {fold}, model {MODEL_NUMBER}, epoch {epoch} **********')
                #The mean calculations are started again at each epoch
                mean_loss = 0
                mean_acc = 0
                iteration = 0
                start_1000_it = time.time() #Variable to record the time spent in 1000 iterations
                #Training loop
                custom_dataset.start_loop(batch_size=batch_size, fold=fold, val_loop=False)
                #The training is performed in batches
                classifier.train()
                for batch in custom_dataset.get_batch_indexes():
                    if iteration % print_rate_1 == 0 and iteration !=0:
                        last = start
                        start = time.time()
                    if iteration == 0:
                        log.save_log(f"Memory usage before reading batch {batch}: {memory_usage()}")
                    X_batch, y_batch = custom_dataset.load_batch()
                    if iteration == 0:
                        log.save_log(f"Memory usage after generating train_loader for batch {batch}: {memory_usage()}")
                    if iteration % print_rate_1 == 0 and iteration !=0:
                        log.save_log(f'Time spent splitting and preparing the training batch: {time.time() - start}, time spent in {print_rate_1} iterations:{time.time() - last}')
                    
                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    correct_pred = (y_pred_tags == y_batch).float()
                    acc = correct_pred.sum() / len(correct_pred)
                    acc = acc * 100

                    mean_loss += loss.item()
                    mean_acc += acc.item()

                    loss.backward()
                    optimizer.step()
                    if iteration % print_rate_2 == 0 and iteration !=0:
                        log.save_log(f'Fold: {fold}, Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, Batch: {batch}, iteration: {iteration}, train loss: {loss.item()}, train acc: {acc.item()}, mean train loss: {mean_loss/(iteration+1)}, mean train acc: {mean_acc/(iteration+1)}')
                        log.save_log(f'Time spent in {print_rate_2} iterations: {time.time() - start_1000_it}')
                        start_1000_it = time.time()
                    iteration += 1
                    df.loc[row] = ["train", fold, MODEL_NUMBER, epoch, batch, loss.item(), acc.item(), mean_loss/iteration, mean_acc/iteration]
                    row += 1

                gc.collect()
                torch.cuda.empty_cache()    # clear cache memory on GPU
                mean_loss = mean_loss/iteration
                mean_acc = mean_acc/iteration
                log.save_log(f'Mean results at the end of epoch {epoch}: mean training loss = {mean_loss}, mean training accuracy = {mean_acc}')
                #Saving the model at the end of each epoch
                start = time.time()
                model_path = os.path.join(args['exp_dir'],
                                        '_fold_' + str(fold) + '_model_' + str(MODEL_NUMBER) +'_epoch_'+ str(epoch) + '_seed_' + str(seed) + '.pth')
                torch.save({'model_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                        model_path)
                log.save_log(f'Time spent saving checkpoint of model {MODEL_NUMBER} and epoch {epoch}: {time.time() - start}')
                
                # Validation loop
                start = time.time()
                custom_dataset.start_loop(batch_size=batch_size, fold=fold, val_loop=True)
                classifier.eval()
                iteration_val = 0
                mean_loss_val = 0
                mean_acc_val = 0
                with torch.no_grad():
                    for batch in custom_dataset.get_batch_indexes():
                        X_batch, y_batch = custom_dataset.load_batch()
                        
                        y_pred = classifier(X_batch)
                        loss = criterion(y_pred, y_batch)
                        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
                        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                        correct_pred = (y_pred_tags == y_batch).float()
                        acc = correct_pred.sum() / len(correct_pred)
                        acc = acc * 100

                        mean_loss_val += loss.item()
                        mean_acc_val += acc.item()

                        iteration_val += 1

                    torch.cuda.empty_cache()    # clear cache memory on GPU
                    #Free memory before loading new chunks
                    gc.collect()
                log.save_log(f'************ Validation Results of fold, {fold}, model {MODEL_NUMBER}, epoch {epoch} ************')
                log.save_log(f'Results: mean validation loss = {mean_loss_val/iteration_val}, mean validation accuracy = {mean_acc_val/iteration_val}')
                df.loc[row] = ["validation", fold, MODEL_NUMBER, epoch, batch, None, None, mean_loss_val/iteration_val, mean_acc_val/iteration_val]
                row += 1
                del batch
                gc.collect()
                torch.cuda.empty_cache()    # clear cache memory on GPU
                log.save_log(f'Time spent in the validation loop: {time.time() - start}')
            df.to_csv(os.path.join(args['exp_dir'], f"results_crossvalidation_folds_{fold_run}_seed_{seed}.csv"))
    log.save_log(f'Time spent in total: {time.time() - startt}')
    hdf5_feature_maps.close()


def generate_results(args, output_path, epoch_selected, fold_run, seeds='sequential', dtype=torch.float16, qualitative=True, generate_metrics=False):
    """To generate qualitative or quantitative results of the 4-fold cross-validation. Its inputs
    are the arguments passed to the execution, the output path to where the
    qualitative results are saved, and the path of where the checkpoints of the
    cross-validation are stored."""

     #Start of execution here
    startt = time.time()

    log.save_log(f'Entered main') #for debugging purposes

    os.makedirs(output_path, exist_ok=True)

    #Choice of the color pallete according to the experiment
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette

    diff_palette = [255, 255, 255,
                    0, 255, 0]

    #Name of the feature maps stored on the disk
    #If you want to store elsewhere, change the varibale to the full pathname
    #Example: feature_maps_filename = "my_path_folder/X_" + args['category'] + ".hdf5"
    if "variation" in args:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + args["variation"] + ".hdf5")
    else:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + ".hdf5")

    #The code below avoids preparing the feature maps multiple times
    if(not os.path.isfile(feature_maps_filename)):
        log.save_log("Preparing data...")
        start = time.time()
        prepare_data(args, palette)
        log.save_log(f'Time spent preparing data: {time.time() - start}')
    log.save_log(f"Loading training feature maps and its respective masks from {feature_maps_filename}")
    hdf5_feature_maps = h5py.File(feature_maps_filename, "r")
    num_data = hdf5_feature_maps["/all_feature_maps_train"].attrs["num_data"]

    #In case you only want to generate the feature maps, add the following line
    #to the json file in experiments folder: "only_gen_data_files": true
    if "only_gen_data_files" in args:
        if(args['only_gen_data_files']):
            sys.exit()

    start = time.time()
    X_mem_map, y_mem_map = hdf5_feature_maps["/all_feature_maps_train"], hdf5_feature_maps["/all_mask_train"]
    log.save_log(f"Training feature maps shape: {X_mem_map.shape}")
    log.save_log(f"Training masks shape: {y_mem_map.shape}")
    classes = args['number_class']
    log.save_log(f"************************* Number of classes: {classes} **************************")
    log.save_log(f"********************* Number of annotated images: {num_data} ********************")
    batch_size = args['batch_size']
    kernel_size = args['kernel_size']
    kernels = args['kernels']
    padding = args['padding']
    folds = args['folds']
    if fold_run >= 0: # If fold_run is positive it means only one fold will be executed
        fold_range = range(fold_run, fold_run + 1)
    else:
        fold_range = range(folds)
    classifier_type = args['classifier_type']

    if generate_metrics:
        total_epochs = range(args['xval_epochs'])
    else:
        total_epochs = range(epoch_selected, epoch_selected+1)

    if seeds == 'sequential':
        seeds = list(range(args['model_num_per_fold_xval_qualitative']))

    save_rate = int(num_data*args['dim'][0]*args['dim'][1]/batch_size/3*(folds-1)/(folds))
    log.save_log(f"save_rate: 1/{save_rate}")

    if generate_metrics:
        df = pd.DataFrame(columns=["mode", "fold", "model", "epoch", "batch", "loss", "accuracy", "mean_loss", "mean_accuracy"])
        row = 0

    for epoch in total_epochs:
        for fold in fold_range:
            fold_pred_masks = []
            masks_list_1 = []
            masks_list_2 = []
            diff_list = []
            diff_list_GT = []
            for MODEL_NUMBER in range(args['model_num_per_fold_xval_qualitative']):
                gc.collect()
                if generate_metrics:
                    mean_loss_val = 0
                    mean_acc_val = 0
                    criterion = nn.CrossEntropyLoss()
                log.save_log(f'Selecting seed {seeds[MODEL_NUMBER]} of fold {fold}, time spent so far: {time.time() - startt}')
                start = time.time()
                #Defines the classifier
                classes = args['number_class']
                if classifier_type == "MLP":
                    classifier = pixel_classifier(numpy_class=(classes), dim=args['dim'][-1])
                    custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size, 
                                                crossval=True, folds=folds, dtype=dtype, conv=False, logger=log)
                elif classifier_type == "convolutional-MLP":
                    classifier = convolutional_pixel_classifier(classes, args['dim'][-1], kernels=kernels, kernel_size=kernel_size)
                    custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size,
                                                    crossval=True, folds=folds, dtype=dtype, logger=log)
                else:
                    assert f"classifier_type {classifier_type} not implemented!"
                #Initial setup of the training
                classifier.init_weights()
                classifier = nn.DataParallel(classifier, device_ids=device_ids).to(device)
                model_path = os.path.join(args['exp_dir'],
                                            '_fold_' + str(fold) + '_model_0'
                                            +'_epoch_'+ str(epoch) + '_seed_' + str(seeds[MODEL_NUMBER]) + '.pth')
                log.save_log(f'Loading checkpoint from {model_path} ...')
                try:
                    checkpoint = torch.load(model_path)
                except:
                    log.save_log(f'WARNING: Not found checkpoint in {model_path}!')
                    log.save_log('Continuing to next checkpoint...')
                    continue

                classifier.load_state_dict(checkpoint['model_state_dict'])

                # Validation loop
                custom_dataset.start_loop(batch_size=batch_size, fold=fold, val_loop=True)
                classifier.eval()
                iteration_val = 0
                pos_counter = 0
                predicted_array = np.empty_like(custom_dataset.get_output_xval(), dtype=int)
                original_shape = predicted_array.shape
                predicted_array = predicted_array.reshape(-1)
                if MODEL_NUMBER == 0:
                    all_GT = np.empty_like(custom_dataset.get_output_xval(), dtype=int)
                    all_GT = all_GT.reshape(-1)
                log.save_log(f'Starting predictions from checkpoint...')
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    with torch.no_grad():
                        # for batch in tqdm(custom_dataset.get_batch_indexes()):
                        for batch in custom_dataset.get_batch_indexes():
                            (X_batch, y_batch), positions = custom_dataset.load_batch(shuffle=False, return_positions=True)          
                            y_pred = classifier(X_batch)
                            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
                            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                            y_pred_tags_np = y_pred_tags.detach().cpu().numpy()
                            if qualitative and epoch==epoch_selected:
                                predicted_array[pos_counter:pos_counter+y_pred_tags_np.shape[0]] = y_pred_tags_np
                                if MODEL_NUMBER == 0:
                                    all_GT[pos_counter:pos_counter+y_pred_tags_np.shape[0]] = y_batch.detach().type(torch.int).cpu().numpy()
                            if generate_metrics:
                                loss = criterion(y_pred, y_batch)
                                correct_pred = (y_pred_tags == y_batch).float()
                                acc = correct_pred.sum() / len(correct_pred)
                                acc = acc * 100
                                mean_loss_val += loss.item()
                                mean_acc_val += acc.item()
                            pos_counter += y_pred_tags.shape[0]
                            iteration_val += 1
                        if qualitative and epoch==epoch_selected:
                            predicted_array = predicted_array.reshape(original_shape)
                            if MODEL_NUMBER == 0:
                                all_GT = all_GT.reshape(original_shape)
                        torch.cuda.empty_cache()
                        gc.collect()
                    if qualitative and epoch==epoch_selected:
                        fold_pred_masks.append(predicted_array)
                    if generate_metrics:
                        log.save_log(f'************ Validation Results of fold, {fold}, model {MODEL_NUMBER}, epoch {epoch} ************')
                        log.save_log(f'Results: mean validation loss = {mean_loss_val/iteration_val}, mean validation accuracy = {mean_acc_val/iteration_val}')
                        df.loc[row] = ["validation", fold, MODEL_NUMBER, epoch, batch, None, None, mean_loss_val/iteration_val, mean_acc_val/iteration_val]
                        row += 1
                    log.save_log(f"Time spent in test loop for checkpoint of fold {fold} and seed {seeds[MODEL_NUMBER]}: {time.time() - start}") 
                except Exception as e:
                    log.save_log(f"batch: {batch}, fold: {fold}, model number: {MODEL_NUMBER}")
                    log.save_log(str(e))
                    raise e 
                del classifier, X_batch, y_batch, y_pred, y_pred_tags, checkpoint, custom_dataset, positions
                gc.collect()
                torch.cuda.empty_cache()
            
            if qualitative and epoch==epoch_selected:
                start_1 = time.time()
                log.save_log(f"Organizing validation masks...")
                for i in range(fold_pred_masks[0].shape[0]):
                    log.save_log(f"Generating masks {i + 1} of {fold_pred_masks[0].shape[0]}...")
                    #Appending the mask of the first network in the fold
                    mask_1 = fold_pred_masks[0][i:i+1,:,:].squeeze()
                    masks_list_1.append(mask_1)
                    #Appending the mask of the second network in the fold
                    mask_2 = fold_pred_masks[1][i:i+1,:,:].squeeze()
                    masks_list_2.append(mask_2)
                    #For the 2 networks in each fold a mask with the difference between
                    #the pixels is generated.
                    diff_mask = np.ma.masked_not_equal(mask_1, mask_2).mask*1
                    # If all the pixels compared are the same np.ma.masked_not_equal() returns ()
                    if diff_mask.shape == ():
                        diff_mask = np.zeros(mask_1.shape)
                    diff_list.append(diff_mask)
                    #Only for the first network in each fold a mask with the difference
                    #between the classified pixels and the ground truth is generated.
                    diff_mask_GT = np.ma.masked_not_equal(all_GT[i:i+1,:,:].squeeze(), mask_1).mask*1
                    # If all the pixels compared are the same np.ma.masked_not_equal() returns ()
                    if diff_mask_GT.shape == ():
                        diff_mask_GT = np.zeros(mask_1.shape)
                    diff_list_GT.append(diff_mask_GT)
                log.save_log(f"Time spent generating for {fold}: {time.time() - start_1}")
                torch.cuda.empty_cache()    # clear cache memory on GPU
                gc.collect()
                log.save_log(f"Processing the masks for the {fold_pred_masks[0].shape[0]} input images...")
                for i in range(fold_pred_masks[0].shape[0]):
                    log.save_log(f"Generating images [{i+1} of {fold_pred_masks[0].shape[0]}]")
                    #Saving the ground truth mask as a .png image
                    GT_color_mask, _ = put_color_mask(all_GT[i], palette)
                    GT_color_mask.save(os.path.join(output_path, f"mask_GT_fold_{fold}_image_{i}.png"))
                    #Saving the first pixel classification mask of the fold as a .png image
                    val_masks_1, _ = put_color_mask(masks_list_1[i], palette)
                    val_masks_1.save(os.path.join(output_path, f"mask_CV_fold_{fold}_image_{i}_seed_{seeds[0]}.png"))
                    #Saving the second pixel classification mask of the fold as a .png image
                    val_masks_2, _ = put_color_mask(masks_list_2[i], palette)
                    val_masks_2.save(os.path.join(output_path, f"mask_CV_fold_{fold}_image_{i}_seed_{seeds[1]}.png"))
                    #Saving the difference of the two pixel classification masks of the
                    #fold as a .png image
                    val_masks_diff, _ = put_color_mask(diff_list[i], diff_palette)
                    val_masks_diff.save(os.path.join(output_path, f"diff_masks_CV_fold_{fold}_image_{i}.png"))
                    #Saving the difference of the first pixel classification mask of the
                    #fold and the ground truth mask as a .png image
                    val_masks_diff_GT, _ = put_color_mask(diff_list_GT[i], diff_palette)
                    val_masks_diff_GT.save(os.path.join(output_path, f"diff_masks_CV_fold_{fold}_image_{i}_and_GT.png"))
    if generate_metrics:
        df.to_csv(os.path.join(args['exp_dir'], f"results_crossvalidation_folds_{'_'.join([str(i) for i in fold_range])}.csv"))
    log.save_log(f"Total time spent generating results from checkpoints {time.time() - startt}")
    hdf5_feature_maps.close()


def main(args, dtype=torch.float16):
    #Start of execution here
    startt = time.time()

    log.save_log(f'Entered main') #for debugging purposes

    #Choice of the color pallete according to the experiment
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    else:
        from utils.data_util import crack_palette as palette

    #Name of the feature maps stored on the disk
    #If you want to store elsewhere, change the varibale to the full pathname
    #Example: feature_maps_filename = "my_path_folder/X_" + args['category'] + ".hdf5"
    if "variation" in args:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + args["variation"] + ".hdf5")
    else:
        feature_maps_filename = os.path.join(os.path.join(args["local_dir"], "feature_maps"), args['category'] + ".hdf5")


    #The code below avoids preparing the feature maps multiple times
    if(not os.path.isfile(feature_maps_filename)):
        #try:
        log.save_log("Preparing data...")
        start = time.time()
        prepare_data(args, palette)
        log.save_log(f'Time spent preparing data: {time.time() - start}')
        #except Exception as e:
        #    if os.path.exists(os.path.join("feature_maps", args['category'] + ".hdf5")):
        #        shutil.rmtree(os.path.join("feature_maps", args['category'] + ".hdf5"))
        #    assert(e)
    log.save_log(f"Loading training feature maps and its respective masks from {feature_maps_filename}")
    hdf5_feature_maps = h5py.File(feature_maps_filename, "r")
    num_data = hdf5_feature_maps["/all_feature_maps_train"].attrs["num_data"]

    #In case you only want to generate the feature maps, add the following line
    #to the json file in experiments folder: "only_gen_data_files": true
    if "only_gen_data_files" in args:
        if(args['only_gen_data_files']):
            sys.exit()

    start = time.time()
    X_mem_map, y_mem_map = hdf5_feature_maps["/all_feature_maps_train"], hdf5_feature_maps["/all_mask_train"]
    log.save_log(f"Training feature maps shape: {X_mem_map.shape}")
    log.save_log(f"Training masks shape: {y_mem_map.shape}")
    classes = args['number_class']
    log.save_log(f"************************* Number of classes: {classes} **************************")
    log.save_log(f"********************* Number of annotated images: {num_data} ********************")
    batch_size = args['batch_size']
    total_epochs = 1
    kernel_size = args['kernel_size']
    kernels = args['kernels']
    padding = args['padding']
    classifier_type = args['classifier_type']

    #The training is executed
    for MODEL_NUMBER in range(args['model_num']):
        stop_sign = False
        gc.collect()
        log.save_log(f'Selecting model {MODEL_NUMBER}, time spent so far: {time.time() - startt}')
        start = time.time()
        #Defines the classifier
        classes = args['number_class']
        if classifier_type == "MLP":
            classifier = pixel_classifier(numpy_class=(classes), dim=args['dim'][-1])
            custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size, dtype=dtype, conv=False)
        elif classifier_type == "convolutional-MLP":
            classifier = convolutional_pixel_classifier(classes, args['dim'][-1], kernels=kernels, kernel_size=kernel_size)
            custom_dataset = CustomDataset(X_mem_map, y_mem_map, padding=padding, kernel_size=kernel_size, dtype=dtype)
        else:
            assert f"classifier_type {classifier_type} not implemented!"
        #Initial setup of the training
        classifier.init_weights()
        if len(device_ids)>1:
            classifier =  nn.DataParallel(classifier, device_ids=device_ids)
        else:
            classifier = classifier.to(device)
        criterion = nn.CrossEntropyLoss()
        if dtype == torch.float16:
            classifier.half()
            optimizer = Adam16(classifier.parameters(), lr=0.001, device=device)
        else:
            optimizer = optim.Adam(classifier.parameters(), lr=0.001)
            
        for epoch in range(100):
            #The mean calculations are started again at each epoch
            mean_loss = 0
            mean_acc = 0
            iteration = 0
            best_loss = 10000000
            start_1000_it = time.time() #Variable to record the time spent in 1000 iterations
            custom_dataset.start_loop(batch_size=batch_size)
            #Training loop
            #The training is performed in chunks
            classifier.train()
            for batch in custom_dataset.get_batch_indexes():
                if iteration%20==0:
                    last = start
                    start = time.time()
                if iteration==0:
                    log.save_log(f"Memory usage before reading batch {batch}: {memory_usage()}")
                X_batch, y_batch = custom_dataset.load_batch()
                gc.collect()
                torch.cuda.empty_cache()    # clear cache memory on GPU
                if iteration==0:
                    log.save_log(f"Memory usage after generating train_loader for batch {batch}: {memory_usage()}")
                if iteration%20==0:
                    log.save_log(f'Time spent splitting and preparing the training batch: {time.time() - start}, time spent in 100 iterations:{time.time() - last}')
                
                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                y_pred_softmax = torch.log_softmax(y_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                correct_pred = (y_pred_tags == y_batch).float()
                acc = correct_pred.sum() / len(correct_pred)
                acc = acc * 100              
                mean_loss += loss.item()
                mean_acc += acc.item()
                loss.backward()
                optimizer.step()
                iteration += 1
                if iteration % 10 == 0:
                    log.save_log(f'Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, Batch: {batch}, iteration: {iteration}, train loss: {loss.item()}, train acc: {acc.item()}, mean train loss: {mean_loss/iteration}, mean train acc: {mean_acc/iteration}')
                    log.save_log(f'Time spent in 10 iterations: {time.time() - start_1000_it}')
                    start_1000_it = time.time()

                if iteration % 500 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                                'model_parts_iter_' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                    log.save_log(f'Save checkpoint, Epoch : {str(epoch)},  Path: {str(model_path)}')
                    torch.save({'model_state_dict': classifier.state_dict()},
                                model_path)
                if epoch >= total_epochs:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 100:
                        stop_sign = True
                        log.save_log(f"*************** Break, Total iters, {iteration} at epoch {str(epoch)} ***************")
                        break
                torch.cuda.empty_cache()    # clear cache memory on GPU

                #Exits batch for loop
                if stop_sign:
                    break
                #Free memory before loading new chunks
                del batch
                gc.collect()
                #Exits Chunk for loop
            mean_loss = mean_loss/iteration
            mean_acc = mean_acc/iteration
            gc.collect()
            log.save_log(f'Mean results at the end of epoch {epoch}: mean training loss = {mean_loss}, mean training accuracy = {mean_acc}')
            #Saving the model at the end of each epoch
            start = time.time()
            model_path = os.path.join(args['exp_dir'],
                                      'model_' + str(MODEL_NUMBER) +'_epoch_'+ str(epoch) +'.pth')
            torch.save({'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_path)
            log.save_log(f'Time spent saving checkpoint of model {MODEL_NUMBER} and epoch {epoch}: {time.time() - start}')
            #Exits Epoch for loop
            if stop_sign:
                break
        #Finishing the execution
        gc.collect()
    log.save_log(f'Time spent in total: {time.time() - startt}')
    hdf5_feature_maps.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)
    parser.add_argument('--crossvalidate', type=bool, default=False)
    parser.add_argument('--cuda_device', type=int,  default=0)
    parser.add_argument('--float32', type=bool,  default=False)
    parser.add_argument('--fold_run', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--generate_qualitative_results', type=bool, default=False)
    parser.add_argument('--generate_metrics', type=bool, default=False)
    parser.add_argument('--results_epoch', type=int)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    
    device = f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu'
    
    path = opts['exp_dir']

    if args.float32:
        dtype = torch.float32
    else:
        dtype = torch.float16

    set_seed(args.seed)
    
    log = LogSaver(save_path=os.path.join(path, "log.txt"))

    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        log.save_log('Experiment folder created at: %s' % (path))
    
    log.save_log(f"Opts: {opts}")

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    
    # If we want to use multiple GPUs a list of them is generated
    # For using only cuda:0, set cuda_device=0, n_gpus=1
    # For using only cuda:1, set cuda_device=1, n_gpus=1
    # For using cuda:0 and cuda:1, set cuda_device=0, n_gpus=2
    device_ids = [i for i in range(args.cuda_device, args.cuda_device+args.n_gpus)]

    if args.generate_data: #In case we want to generate images
        generate_data(opts, args.resume, args.num_sample, args.start_step, vis=args.save_vis, dtype=dtype)
    elif args.crossvalidate:
        cross_val(opts, args.fold_run, args.seed, dtype=dtype)
    elif args.generate_qualitative_results and args.generate_metrics:
        generate_results(opts, args.output_dir, args.results_epoch, args.fold_run, dtype=dtype, generate_metrics=True)
    elif args.generate_qualitative_results:
        generate_results(opts, args.output_dir, args.results_epoch, args.fold_run, dtype=dtype)
    elif args.generate_metrics:
        generate_results(opts, args.output_dir, args.results_epoch, args.fold_run, dtype=dtype, generate_metrics=True, qualitative=False)
    else: #In case we want to train the pixel_classifier
        main(opts, dtype=dtype)
