import os
import logging
from datetime import datetime
import sys
import time
import random
import pdb
import glob
import pickle
import cv2
import numpy as np
# laplace function
X1 = 0
Y1 = 1
WIDTH = 2
HEIGHT = 3

laplace = lambda a,b: np.random.laplace(a, b, 1)[0]

data_set = 'imagedata++'
gt_set = 'alov300++_rectangleAnnotation_full'

data_folders = sorted([os.path.join(data_set,clip) for clip in os.listdir(data_set) if not clip.endswith('.DS_Store')])
gt_folders = sorted([os.path.join(gt_set,clip) for clip in os.listdir(gt_set) if not clip.endswith('.DS_Store')])
clips = []
for data_folder in data_folders:
    clips += sorted([os.path.join(data_folder,clip) for clip in os.listdir(data_folder) if not clip.endswith('.DS_Store')])

gts = []
for gt_folder in gt_folders:
    gts += sorted([os.path.join(gt_folder,clip) for clip in os.listdir(gt_folder) if not clip.endswith('.DS_Store')])

box_track = []
num_clips = len(clips)
for i in range(num_clips):
    gt_file = open(gts[i])
    content = gt_file.read().splitlines()
    gt_file.close()
    prev_file = None
    prev_box = None
    for line in content:
        split_list = line.split()
        curr_file = os.path.join(clips[i],split_list[0].zfill(8)+'.jpg')
        [x1,y1,x2,y2] = [float(split_list[3]),float(split_list[2]),
                        float(split_list[1]),float(split_list[6])]
        x1 = min(float(split_list[1]),float(split_list[3]),float(split_list[5]),float(split_list[7]))
        x2 = max(float(split_list[1]),float(split_list[3]),float(split_list[5]),float(split_list[7]))
        y1 = min(float(split_list[2]),float(split_list[4]),float(split_list[6]),float(split_list[8]))
        y2 = max(float(split_list[2]),float(split_list[4]),float(split_list[6]),float(split_list[8]))
        curr_box = [x1,y1,x2-x1,y2-y1]
        if prev_box != None:
            box_track += [[prev_file,curr_file,
            prev_box[0],prev_box[1],prev_box[2],prev_box[3],
            curr_box[0],curr_box[1],curr_box[2],curr_box[3]]]
        prev_box = curr_box
        prev_file = curr_file

center_b = 0.2
length_b = 1/15

target_folder = "train/target/"
search_folder = "train/searching/"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
if not os.path.exists(search_folder):
    os.makedirs(search_folder)


train_list = []
print("**********start generating training pictures***********")
for idx,box in enumerate(box_track):
    prev_frame = cv2.imread(box[0])
    next_frame = cv2.imread(box[1])
    prev_box = box[2:6]
    prev_center = [prev_box[X1]+prev_box[WIDTH]/2,prev_box[Y1]+prev_box[HEIGHT]/2]
    next_box = box[6:10]
    next_center = [next_box[X1]+next_box[WIDTH]/2,next_box[Y1]+next_box[HEIGHT]/2]
    center_bias = laplace(0,0.2)
    search_center = [prev_center[X1]+laplace(0,center_b)*prev_box[WIDTH],\
                        prev_center[Y1]+laplace(0,center_b)*prev_box[HEIGHT]]
    while(True):
        gamma_w = laplace(1,length_b)
        if (gamma_w < 1.4 and gamma_w > 0.6):
            break
    while(True):
        gamma_h = laplace(1,length_b)
        if (gamma_h < 1.4 and gamma_h > 0.6):
            break
    search_width = prev_box[WIDTH] * gamma_w * 2
    search_height = prev_box[HEIGHT] * gamma_h * 2
    # search_box: [x1,y1,x2,y2]
    # search_width, search_height
    search_box = [search_center[X1] - search_width/2, search_center[Y1] - search_height/2,\
                    search_center[X1] + search_width/2, search_center[Y1] + search_height/2]
    orig_box = prev_box[:2] + [prev_box[0]+prev_box[2],prev_box[1]+prev_box[3]] # [x1,y1,x2,y2]
    


    # print("original:%s"%(orig_box),"new_search:%s"%(search_box[:4]))

    # deal with bounding exceeding problem:
    [x1, y1, x2, y2] = search_box
    [x1, y1, x2, y2] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    img_height, img_width, _ = prev_frame.shape
    new_search_box = [max(x1-1,0),max(y1-1,0),min(x2-1,img_width-1),min(y2-1,img_height-1)]

    # target_box : [x1,y1,x2,y2], double size of the bounding box of prev_frame
    target_width = prev_box[2]
    target_height = prev_box[3]
    [x1, y1, x2, y2] = [int(round(orig_box[0] - target_width/2)), int(round(orig_box[1] - target_height/2)),
                int(round(orig_box[2] + target_width/2)), int(round(orig_box[3] + target_height/2))]

    # [x1, y1, x2, y2] = [int(round(prev_box[X1]-1)),int(round(prev_box[Y1]-1)),\
    #                     int(round(prev_box[X1]+prev_box[WIDTH]-1)),int(round(prev_box[Y1]+prev_box[HEIGHT]-1))]
    new_target_box = [max(x1-1,0),max(y1-1,0),min(x2-1,img_width-1),min(y2-1,img_height-1)]

    # the images need to output
    prev_target_img = prev_frame[new_target_box[1]:new_target_box[3]+1,new_target_box[0]:new_target_box[2]+1]
    search_img = next_frame[new_search_box[1]:new_search_box[3]+1,new_search_box[0]:new_search_box[2]+1]



    [new_search_width, new_search_height] = [new_search_box[2] - new_search_box[0], new_search_box[3] - new_search_box[1]]
    if new_search_width < 10 or new_search_height < 10:
        # remove searching boxes which are too small
        continue
    # gt_box = [x1,y1,x2,y2]
    gt_box = [(next_box[X1]-new_search_box[0]-1)/new_search_width,(next_box[Y1]-new_search_box[1]-1)/new_search_height,\
                    (next_box[X1]-new_search_box[0]-1+next_box[WIDTH])/new_search_width,\
                    (next_box[Y1]-new_search_box[1]-1+next_box[HEIGHT])/new_search_height]

    # gt_box = [max(gt_box[0],0),max(gt_box[1],0),min(gt_box[2],1),min(gt_box[3],1)]



    filename = str(idx).zfill(6)+".jpg"
    cv2.imwrite(os.path.join(target_folder,filename),prev_target_img)
    cv2.imwrite(os.path.join(search_folder,filename),search_img)

    train_list.append(os.path.join(target_folder,filename)+","+os.path.join(search_folder,filename)+","\
                        + str(gt_box[0]) +","+ str(gt_box[1]) +","+ str(gt_box[2]) +","+ str(gt_box[3]))

f = open("train.txt","w+")
for item in train_list:
    f.write(item+'\n')
f.close()