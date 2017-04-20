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
laplace = lambda a,b: np.random.laplace(a, b, 1)[0]


data_set = ['MOT17Det/train', '2DMOT2015/train']
clips = []
for data_folder in data_set:
    clips += [os.path.join(data_folder,clip) for clip in os.listdir(data_folder) if not clip.endswith('.DS_Store')]
# now each clip is the folder of clip, it has three folders:
# det, gt, imgs
# the bounding box are 1-based, the left most one is idx 1
# gt.txt file: <frame> <target> <x1> <y1> <width> <height> <...>

X1 = 0
Y1 = 1
WIDTH = 2
HEIGHT = 3

train_info = {}
for idx,clip_path in enumerate(clips):
    # print(clip_path)
    print(idx)
    gt_txt = os.path.join(clip_path,"gt/gt.txt")
    gt_file = open(gt_txt, 'r')
    content = gt_file.read().splitlines()
    content_split = []
    for line in content:
        split_list = line.split(',')
        if (clip_path.startswith('MOT17Det') and split_list[7] != '1' or 
            clip_path.startswith('2DMOT2015') and split_list[6] != '1'):
            continue
        tmp_list = split_list[:6]
        tmp_list = [tmp_list[0].zfill(6)+'.jpg', int(tmp_list[1]), float(tmp_list[2]),
                     float(tmp_list[3]), float(tmp_list[4]), float(tmp_list[5])]
        tmp_list[0], tmp_list[1] = tmp_list[1], tmp_list[0]
        content_split += [tmp_list]
    content_split_sort = sorted(content_split)
    train_info[os.path.join(clip_path,'img1')] = content_split_sort

# choose 1 frame from every 5 frames
train_info_small = {}
for clip_path, boxes in train_info.items():
    train_info_small[clip_path] = [box for idx,box in enumerate(boxes) if idx % 3 == 1]

box_track = []
for clip_path, boxes in train_info_small.items():
    prev_box = None
    for curr_box in boxes:
        if not prev_box:
            prev_box = curr_box
            continue
        # now it is not the first image
        print(curr_box[0],prev_box[0])
        if curr_box[0] == prev_box[0] and int(curr_box[1][:6]) - 3 == int(prev_box[1][:6]):
            # if it is the same video and it is two consecutive frames
            new_line_list = [os.path.join(clip_path,prev_box[1]),os.path.join(clip_path,curr_box[1])] \
                            + prev_box[2:] + curr_box[2:]
            box_track += [new_line_list]
        prev_box = curr_box

# pickle.dump(box_track, open("trainset.p", "wb"))

# f = open('test.txt', 'w+')
# for item in box_track:
#     f.write(str(item) + '\n')

# use this to restore
# box_track_load = pickle.load(open("trainset.p", "rb"))

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

    # target_box : [x1,y1,x2,y2], it is the bounding box of prev_frame
    [x1, y1, x2, y2] = [int(round(prev_box[X1]-1)),int(round(prev_box[Y1]-1)),\
                        int(round(prev_box[X1]+prev_box[WIDTH]-1)),int(round(prev_box[Y1]+prev_box[HEIGHT]-1))]
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





