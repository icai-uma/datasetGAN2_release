"""
This code was generated based on 'DatasetGAN: Efficient Labeled Data Factory
with Minimal Human Effort' publication, available at:
https://nv-tlabs.github.io/datasetGAN/

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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152 of DatasetGAN repository

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from utils.datasetgan_utils import multi_acc, colorize_mask, latent_to_image, oht_to_scalar, Interpolate
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
import argparse
import cv2
import pandas as pd
import time
from models.stylegan1 import G_mapping,Truncation,G_synthesis
from utils.utils import memory_usage, put_color_mask

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
            assert "Not implemented!"

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


def generate_qualitative_results(args, output_path, checkpoint_path):
    """To generate qualitative results of the 4-fold cross-validation. Its inputs
    are the arguments passed to the execution, the output path to where the
    qualitative results are saved, and the path of where the checkpoints of the
    cross-validation are stored."""
    start_0 = time.time()

    dim0 = args['dim'][0]
    dim1 = args['dim'][1]

    #Pallete to show the difference between approaches
    diff_palette = [255, 255, 255,
                    0, 255, 0]

    #Choosing the palette colors
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

    #Creating a folder for comparison if it does not exists
    if os.path.exists(output_path):
        pass
    else:
        os.system('mkdir -p %s' % (output_path))
        save_log('Folder created at: %s' % (output_path))

    name_x = "X_" + args['category'] + ".npy"
    name_y = "y_" + args['category'] + ".npy"

    #For each fold the validation images are generated
    for fold in range(4):
        all_GT = []
        masks_list_1 = []
        masks_list_2 = []
        diff_list = []
        diff_list_GT = []
        l_pred_masks = []
        save_log(f"Loading data from fold {fold+1} of 4...")
        X_mem_map, y_mem_map = load_X_y_as_mem_map(name_x, name_y)
        save_log(f"X shape: {X_mem_map.shape}")
        save_log(f"y shape: {y_mem_map.shape}")
        num_data = int(y_mem_map.shape[0]/(dim1)/(dim0))
        chunk_list_train, chunk_list_validation = gen_cross_val_chunk_list(fold, num_data)
        save_log(f"Number of images selected for validation: {len(chunk_list_validation)}")
        batch_size = args['batch_size']
        max_label = args['max_label']
        save_log(f"The dataset has a total of {args['number_class']} different classes")
        #Saving to the output folder the ground truth masks available in the feature maps
        for chunk in chunk_list_validation:
            save_log(f"Saving groud truth mask {chunk}...")
            name = os.path.join(output_path, f"GT_mask_{chunk}.npy")
            _, y_local  = get_chunk_from_memmap(X_mem_map, y_mem_map, chunk, (dim1)*(dim0))
            np.save(name, y_local.reshape((dim0, dim1)))
            all_GT.append(y_local.reshape((dim0, dim1)))
        gc.collect()
        start_1 = time.time()
        #For each checkpoint generated by a network within this fold
        for MODEL_NUMBER in range(fold*2,fold*2+2):
            start_2 = time.time()
            pred_mask = []
            classifier = pixel_classifier(numpy_class=args['number_class'],
                                          dim=args['dim'][-1],
                                          dropout_prob_1=args['dropout_1'],
                                          dropout_prob_2=args['dropout_2']
                                          )

            classifier =  nn.DataParallel(classifier, device_ids=device_ids).cuda()
            save_log(f"Loading checkpoint {MODEL_NUMBER+1} of 8...")
            #In case the results to be plotted are not from the last epoch,
            #indicate in the json inside experiments folder from wich epoch to
            #load the checkpoints
            if "epoch_best" in args:
                epoch_best = args['epoch_best']
                checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) +'_epoch_'+ str(epoch_best) +'.pth'))
            else:
                checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) +'_epoch_14' +'.pth'))
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()

            #For each image in the feature maps, masks are generated
            for chunk in chunk_list_validation:
                start = time.time()
                save_log(f"Memory usage before reading validation chunk {chunk}: {memory_usage()}")
                X, y = get_chunk_from_memmap(X_mem_map, y_mem_map, chunk, (dim1)*(dim0))
                validation_data = trainData(torch.FloatTensor(X), torch.FloatTensor(y))
                gc.collect()
                save_log(f"Memory usage after generating validation_data: {memory_usage()}")
                save_log(f'Time spent splitting and preparing the test dataset: {time.time() - start}')
                #creating a dataloader for the validation image
                validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, num_workers=len(device_ids))
                gc.collect()
                save_log(f"Classifying the pixels...")
                len_validation_loader = len(validation_loader)
                count = 0
                classifier.eval()
                with torch.no_grad():
                    for X_batch, y_batch in validation_loader: #Here the test images are used through validation_loader
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_batch = y_batch.type(torch.long)
                        y_pred = classifier(X_batch) #predicts the batch -> dim[max_label, batch_size]
                        try:
                            y_pred = oht_to_scalar(y_pred.squeeze()) #Merges the several classes in a single tensor
                        except Exception as e:
                            print(f"ERROR: {e}")
                            print(f"Chunk: {chunk}, count: {count}, y_pred.shape: {y_pred.shape}, y_pred.squeeze().shape: {y_pred.squeeze().shape}")
                            torch.set_printoptions(edgeitems=3)
                            print(y_pred)
                            print(y_pred.squeeze())
                            sys.exit()
                        pred_mask.append(y_pred.cpu().detach().numpy()) #Appends the batch as a numpy array
                        count += 1

            #Transforms the list of numpy array in a single numpy array
            #unflattened and appends it ina list.
            l_pred_masks.append(np.concatenate(pred_mask).ravel().reshape((-1, dim0, dim1)))
            gc.collect()
            save_log(f"Time spent in test loop for checkpoint {MODEL_NUMBER+1}: {time.time() - start_2}")

        save_log(f"Organizing validation masks...")
        for i in range(l_pred_masks[0].shape[0]):
            save_log(f"Generating masks {i + 1} of {l_pred_masks[0].shape[0]}...")
            #Appending the mask of the first network in the fold
            mask_1 = l_pred_masks[0][i]
            masks_list_1.append(mask_1)
            #Appending the mask of the second network in the fold
            mask_2 = l_pred_masks[1][i]
            masks_list_2.append(mask_2)
            #For the 2 networks in each fold a mask with the difference between
            #the pixels is generated.
            diff_list.append(np.ma.masked_not_equal(mask_1, mask_2).mask*1)
            #Only for the first network in each fold a mask with the difference
            #between the classified pixels and the ground truth is generated.
            diff_list_GT.append(np.ma.masked_not_equal(all_GT[i], mask_1).mask*1)
        save_log(f"Time spent generating for {fold+1} of 4: {time.time() - start_1}")
        torch.cuda.empty_cache()    # clear cache memory on GPU
        gc.collect()
        save_log(f"Processing the masks for the {l_pred_masks[0].shape[0]} input images...")
        for i in range(l_pred_masks[0].shape[0]):
            save_log(f"Generating images [{i+1} of {l_pred_masks[0].shape[0]}]")
            #Saving the ground truth mask as a .png image
            GT_color_mask, _ = put_color_mask(all_GT[i], palette)
            GT_color_mask.save(os.path.join(output_path, "mask_GT_" + str(fold*len(chunk_list_validation)+i) + '.png'))
            #Saving the first pixel classification mask of the fold as a .png image
            val_masks_1, _ = put_color_mask(masks_list_1[i], palette)
            val_masks_1.save(os.path.join(output_path, f"mask_CV{fold*2+1}_" + str(fold*len(chunk_list_validation)+i) + '.png'))
            #Saving the second pixel classification mask of the fold as a .png image
            val_masks_2, _ = put_color_mask(masks_list_2[i], palette)
            val_masks_2.save(os.path.join(output_path, f"mask_CV{fold*2+2}_" + str(fold*len(chunk_list_validation)+i) + '.png'))
            #Saving the difference of the two pixel classification masks of the
            #fold as a .png image
            val_masks_diff, _ = put_color_mask(diff_list[i], diff_palette)
            val_masks_diff.save(os.path.join(output_path, f"diff_masks_CV{fold*2+1}_CV{fold*2+2}_" + str(fold*len(chunk_list_validation)+i) + '.png'))
            #Saving the difference of the first pixel classification mask of the
            #fold and the ground truth mask as a .png image
            val_masks_diff_GT, _ = put_color_mask(diff_list_GT[i], diff_palette)
            val_masks_diff_GT.save(os.path.join(output_path, f"diff_masks_CV{fold*2+1}_GT_" + str(fold*len(chunk_list_validation)+i) + '.png'))

    save_log(f"Total time spent generating images from checkpoints {time.time() - start_0}")

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

def get_chunk_from_memmap(X, y, chunk_index, chunk_size):
    """Function to load chunks of the feature maps. Its inputs are the features
    (X) and masks (y) as memory maps, the chunk index, and the chunk size (number
     of pixels). It returns a tuple containing the chunk of features and chunk of
     masks"""
    save_log(f"Getting chunk from X and y memmaps. Chunk index: {chunk_index}, chunk size: {chunk_size} pixels")
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size
    return X[start_index:end_index], y[start_index:end_index]

def gen_cross_val_chunk_list(fold, num_chunks):
    """Function to generate train and validation chunks indexes taking as input
    on the fold and the number of chunks. It returns the list of image indexes
    to be used for training and the list of image indexes to be used for validation."""
    chunk_list_train = []
    chunk_list_validation = []
    count = 0
    for i in range(4):
        for j in range(int(num_chunks/4)):
            if i == fold: chunk_list_validation.append(count)
            else: chunk_list_train.append(count)
            count+=1
    return chunk_list_train, chunk_list_validation

def save_log(data, file="log.txt", print_data=True):
    """Function print and store the logs."""
    if(print_data):
        print(str(data))
    with open(os.path.join(file), 'a') as f:
        f.write(data)
        f.write("\n")

def main(args, fold, save_inprogress_ckp = False):
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

    #Dataframe that will store the train and test results
    performance_df = pd.DataFrame(columns=["params","network", "epoch", "iteration", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])

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
    #The variable fold stores which fold is ran and is passed as an
    #argument when the code is executed
    chunk_list_train, chunk_list_validation = gen_cross_val_chunk_list(fold, num_data)
    max_label = args['max_label'] #Index of the last lable
    save_log(f" ************************* max_label {str(max_label)} **************************")
    save_log(f"*********** Current number of images used in training {str(int(num_data/4)*4)} ************") # it makes sure it is a multiple of 4
    batch_size = args['batch_size']
    total_epochs = 15
    #The 4-fold cross-validation is executed
    for MODEL_NUMBER in range(fold*2,fold*2+2): #for each network in the fold...
        gc.collect()
        save_log(f'Selecting model {MODEL_NUMBER}, time spent so far: {time.time() - startt}')
        start = time.time()

        #Defines the classifier
        classifier = pixel_classifier(numpy_class=args['number_class'],
                                      dim=args['dim'][-1],
                                      dropout_prob_1=args['dropout_prob_1'],
                                      dropout_prob_2=args['dropout_prob_2']
                                      )
        #Recording the layers of the classifier in the top of the DataFrame for debugging purposes
        for name in classifier.layers: #Each layer of the pixel_classifier is stored
            performance_df = pd.concat([performance_df, pd.DataFrame(np.array([[str(name), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]), columns=["params", "network", "epoch", "iteration", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])], ignore_index=True)
            save_log(str(name))
        #Initial setup of the training
        classifier.init_weights()
        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        for epoch in range(total_epochs):
            #The mean calculations are started again at each epoch
            mean_loss = 0
            mean_acc = 0
            iteration = 0
            split_subset_train = 8
            start_1000_it = time.time() #Variable to record the time spent in 1000 iterations
            #The training is performed in chunks
            for chunk in chunk_list_train:
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

                    save_log(f'Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, mean loss: {mean_loss/iteration}, mean accuracy: {mean_acc/iteration}')
                    save_log(f'Time spent training model {MODEL_NUMBER}, chunk {chunk} subset {subset}: {time.time() - start}')
                #Free memory before loading new chunks
                del train_subset
                del train_loader
                del train_data
                gc.collect()
            #Mean train loss and accuracy are calculated for each epoch and each model
            mean_loss = mean_loss/iteration
            mean_acc = mean_acc/iteration
            save_log(f"************* Starting Test loop with test dataset, epoch: {epoch} ***************")
            #Preparation for test loop at the end of each epoch
            split_subset_test = split_subset_train
            mean_test_loss = 0
            mean_test_acc = 0
            iteration_val = 0
            start_1000_it = time.time()
            starttest = start_1000_it
            #The test is performed in chunks
            for chunk in chunk_list_validation:
                start = time.time()
                save_log(f"Memory usage before reading test chunk {chunk}: {memory_usage()}")
                X_local, y_local  = get_chunk_from_memmap(X_mem_map, y_mem_map, chunk, (width)*(height))
                validation_data = trainData(torch.FloatTensor(X_local),
                                       torch.FloatTensor(y_local))
                gc.collect()
                save_log(f"Memory usage after generating validation_data: {memory_usage()}")
                save_log(f'Time spent splitting and preparing the test data: {time.time() - start}')
                #The test is performed in chunks, where each chunk is subdivided in dataloaders
                end_position = 0
                for subset in range(split_subset_test):
                    start = time.time()
                    start_position = int(end_position) #Start position of the subset
                    #End position of the subset
                    if (subset == split_subset_test - 1): end_position = len(validation_data)
                    else: end_position += int(len(validation_data)/split_subset_test) #To make sure the int() casting will not make end_position end before the end
                    test_subset = Subset(validation_data, list(range(start_position, end_position)))
                    validation_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=True,  num_workers=len(device_ids))
                    gc.collect()

                    #Test loop
                    classifier.eval()
                    with torch.no_grad():
                        start = time.time()
                        for X_batch, y_batch in validation_loader: #Here the test images are used through validation_loader
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            y_batch = y_batch.type(torch.long)
                            y_batch = y_batch.type(torch.long)
                            y_pred = classifier(X_batch)
                            test_loss = criterion(y_pred, y_batch).item()
                            test_acc = multi_acc(y_pred, y_batch).item()
                            mean_test_loss += test_loss
                            mean_test_acc += test_acc
                            iteration_val += 1
                            if iteration_val % 10000 == 0: #Plot results at each 1000 iterations
                                save_log(f'Model: {MODEL_NUMBER}, Epoch: {str(epoch)}, Chunk: {chunk}, Subset {subset}, iteration: {iteration_val}, test loss: {test_loss}, test acc: {test_acc}, mean test loss: {mean_test_loss/iteration_val}, mean test acc: {mean_test_acc/iteration_val}')
                                gc.collect()
                                save_log(f'Time spent in 10000 test iteration: {time.time() - start_1000_it}')
                                start_1000_it = time.time()
                    save_log(f'Epoch :  {str(epoch)}, model: {MODEL_NUMBER}, test iteration: {iteration_val} , mean test loss: {mean_test_loss/iteration_val}, mean test acc: {mean_test_acc/iteration_val}')
                #Free memory before loading new chunks
                del test_subset
                del validation_loader
                del validation_data
                gc.collect()
            #Mean test loss and accuracy are calculated
            mean_test_loss = mean_test_loss/iteration_val
            mean_test_acc = mean_test_acc/iteration_val
            #Results of the current epoch are added to the dataframe
            performance_df = pd.concat([performance_df, pd.DataFrame(np.array([[np.nan, MODEL_NUMBER, epoch, iteration, mean_loss, mean_acc, mean_test_loss, mean_test_acc]]), columns=["params", "network", "epoch", "iteration", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])], ignore_index=True)
            gc.collect()
            save_log(f'Time spent in the test loop for epoch {epoch}: {time.time() - starttest}')
            #Saving the model at the end of each epoch
            start = time.time()
            model_path = os.path.join(args['exp_dir'],
                                      'model_' + str(MODEL_NUMBER) +'_epoch_'+ str(epoch) +'.pth')
            torch.save({'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_path)
            save_log(f'Time spent saving checkpoint of model {MODEL_NUMBER} and epoch {epoch}: {time.time() - start}')
    #Finishing the execution
    gc.collect()
    torch.cuda.empty_cache()    # clear cache memory on GPU
    #At the end of the execution the dataframe with the mean train/test loss and accuracy for each epoch is saved
    name_file = "nn_performance_" + args['category'] +"_"+ args['test_name'] +"_"+ str(fold) + ".xlsx"
    performance_df.to_excel(name_file)
    save_log(f'Time spent in total: {time.time() - startt}')

if __name__ == '__main__':
    save_log('Started execution')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--classifier_param', type=str, default = "experiments/nn_classifier.json")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--generate_qualitative_results', type=bool, default=False)
    parser.add_argument('--output_dir', type=str,  default="")
    args = parser.parse_args()

    if args.output_dir == "": args.output_dir = args.exp_dir

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

    if args.generate_qualitative_results:
        device_ids = [i for i in range(args.n_processes)] #If we want to use multiple GPUs a list of them is generated
        generate_qualitative_results(opts, args.output_dir, args.exp_dir)
    else: #In case we want to train the pixel_classifier
        device_ids = [i for i in range(args.n_processes)] #If we want to use multiple GPUs a list of them is generated
        main(opts, args.fold)
