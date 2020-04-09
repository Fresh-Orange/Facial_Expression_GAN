#-*- coding:utf-8 -*-

from __future__ import print_function

import os, sys
import cv2
import json
from tqdm import tqdm
from body_demo import VideoReader
import numpy as np
from scipy.ndimage.filters import gaussian_filter


# load annotations of all videos
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print("load %d video annotations totally." % len(data))

    vid_anns = dict()
    for anns in data:
        name = anns['video']
        path = anns['video_path']
        vid_anns[name] = {'path': path}
        for ann in anns['annotations']:
            idx = ann['index']
            keypoints = ann['keypoint']
            bbox = ann['bbox']
            vid_anns[name][idx] = [bbox, keypoints]

    return vid_anns

def crop_face(img, bbox, keypoint):
    flags = list()
    points = list()

    # can not detect face in some images
    if len(bbox) == 0:
        return None

    # draw bbox
    x, y, w, h = [int(v) for v in bbox]
    crop_img = img[y:y+h, x:x+w]
    return crop_img


def crop_big_face(img, bbox, keypoint):
    flags = list()
    points = list()

    # can not detect face in some images
    if len(bbox) == 0:
        return None

    img = cv2.copyMakeBorder(img, 500, 500, 500, 500, cv2.BORDER_REPLICATE)  # 背景的边缘扩展

    # draw bbox
    x, y, w, h = [int(v) for v in bbox]
    crop_img = img[y:y+h, x:x+w]
    return crop_img

def draw_bbox_keypoints(img, bbox, keypoint):
    flags = list()
    points = list()

    # can not detect face in some images
    if len(bbox) == 0:
        return None

    points_image = np.zeros((224,224,3), np.uint8)

    # draw bbox
    bx, by, bw, bh = [int(v) for v in bbox]
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # draw points
    for i in range(0, len(keypoint), 3):
        x, y, flag = [int(k) for k in keypoint[i: i+3]]
        flags.append(flag)
        x = int((x - bx) / bw * 224)
        y = int((y - by) / bh * 224)
        points.append((x, y))
        #points.append( [int((x - bx) / bw * 224), int((y - by) / bh * 224)] )
        if flag == 0:      # keypoint not exist
            continue
        elif flag == 1:    # keypoint exist but invisible
            cv2.circle(points_image, (x, y), 2, (0, 0, 255), -1)
        elif flag == 2:    # keypoint exist and visible
            cv2.circle(points_image, (x, y), 2, (0, 255, 0), -1)
        else:
            raise ValueError("flag of keypoint must be 0, 1, or 2.")

    # draw links
    # face_keypoint_links = get_face_landmark_links()
    # for start, end in face_keypoint_links:
    #     if flags[start] == 0 or flags[end] == 0:
    #         continue
    #     else:
    #         x1, y1 = points[start]
    #         x2, y2 = points[end]
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    return points_image #cv2.GaussianBlur(points_image, ksize=(7, 7), sigmaX=2)


def crop_video(anno):
    vid_path = anno['path']
    vid = VideoReader(vid_path)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # size = (vid.width, vid.height)
    # vid_writer = cv2.VideoWriter(out, fourcc, vid.fps, size)

    video_name = os.path.basename(vid_path).split(".")[0]
    # os.mkdir("crop_face/{}".format(video_name))
    os.makedirs("points_face/{}".format(video_name), exist_ok=True)
    #os.makedirs("points_face_gaussion/{}".format(video_name), exist_ok=True)

    for idx, frm in tqdm(enumerate(vid.read())):
        bbox, keypoint = anno[idx+1]
        # crop_frm = crop_face(frm, bbox, keypoint)
        # if crop_frm is not None:
        #     cv2.imwrite("crop_face/{}/{}.jpg".format(video_name, idx), crop_frm)

        points_frm = draw_bbox_keypoints(frm, bbox, keypoint)
        if points_frm is not None:
            cv2.imwrite("points_face/{}/{}.jpg".format(video_name, idx), points_frm)
    # vid_writer.release()


def main():
    json_path = "face/face_keypoint_train_part2.json"
    vid_annos = load_json(json_path)

    # vid_path = 'face/train/part1/12008.mp4'
    # assert os.path.exists(vid_path)
    # vid_name = os.path.basename(vid_path)
    # print(vid_name)
    # anno = vid_annos[vid_name]
    for key, value in vid_annos.items():
        anno = value
        crop_video(anno)




if __name__=="__main__":
    main()



