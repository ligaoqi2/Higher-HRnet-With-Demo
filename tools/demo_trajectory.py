# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import numpy as np

import glob
import pickle

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.vis import add_joints
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
import datetime
import cv2

from tqdm import tqdm
from video_preprocess import video_preprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def load_higher__hrnet_model():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)
    return transforms, model, parser


def higherhrnet_pose_estimation(vid_file, transforms, model, parser):
    cap = cv2.VideoCapture(vid_file)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    poses = []
    for _ in tqdm(range(int(count)), desc="Processing: "):
        ret, image = cap.read()

        a = datetime.datetime.now()

        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            # Filter for highest score
            index = 0
            for i in range(len(scores) - 1):
                if scores[i] > scores[i + 1]:
                    index = i
                else:
                    index = i + 1

            # print("The highest score index is {}".format(index))

            grouped_filtered = []
            grouped_filtered.append(grouped[0][index, :, :].reshape(1, 17, -1))

            final_results = get_final_preds(
                grouped_filtered, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        pose = final_results[0][:, 0: 2]
        poses.append(pose)
        # Display the resulting frame
        # for person in final_results:
        #     color = np.random.randint(0, 255, size=3)
        #     color = [int(i) for i in color]
        #     add_joints(image, person, color, test_dataset.name, cfg.TEST.DETECTION_THRESHOLD)
        #
        # image = cv2.putText(image, "{:.2f} ms / frame".format(inf_time), (40, 40),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    poses_array = np.array(poses)
    return poses_array


def generate_higherhrnet_pkl(mp4_path):
    transforms, model, parser = load_higher__hrnet_model()
    mp4_files = sorted(glob.glob(mp4_path))
    i = 1
    whole_data = {}
    for mp4_file in mp4_files:
        print("正在处理第 {} 个视频".format(i))
        person_data = {}
        person_name = mp4_file.split("/")[-3]
        poses = higherhrnet_pose_estimation(mp4_file, transforms, model, parser)

        person_data["wrist"] = poses[:, 9, :]
        person_data["ankle"] = poses[:, 15, :]
        whole_data[person_name] = person_data
        i += 1
    with open("/home/ligaoqi/projects/python_projects/openpose/trajectory_paper/HigherHRNet_trajectory.pkl", "wb") as f:
        pickle.dump(whole_data, f)
    print("ok")


if __name__ == '__main__':
    video_path = "/home/ligaoqi/papers/dataset/*/inf/inf_no.mp4"
    generate_higherhrnet_pkl(mp4_path=video_path)
