#-*- coding:utf-8 -*-

from __future__ import print_function

import os, sys
import cv2
import json
from tqdm import tqdm



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


def draw_bbox_keypoints(img, bbox, keypoint):
    flags = list()
    points = list()

    # can not detect face in some images
    if len(bbox) == 0:
        return img

    # draw bbox
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # draw points
    for i in range(0, len(keypoint), 3):
        x, y, flag = [int(k) for k in keypoint[i: i+3]]
        flags.append( flag )
        points.append( [x, y] )
        if flag == 0:      # keypoint not exist
            continue
        elif flag == 1:    # keypoint exist but invisible
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        elif flag == 2:    # keypoint exist and visible
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        else:
            raise ValueError("flag of keypoint must be 0, 1, or 2.")

    return img


def render_video(anno, out):
    vid_path = anno['path']
    vid = VideoReader(vid_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (vid.width, vid.height)
    vid_writer = cv2.VideoWriter(out, fourcc, vid.fps, size)

    for idx, frm in tqdm(enumerate(vid.read())):
        bbox, keypoint = anno[idx+1]
        draw_frm = draw_bbox_keypoints(frm, bbox, keypoint)
        vid_writer.write(draw_frm)
    vid_writer.release()


def main():
    json_path = r"D:\AI_data\expression_transfer\face1\face\face_keypoint_train_part1.json"
    vid_annos = load_json(json_path)

    vid_path = r'D:\AI_data\expression_transfer\face1\face\train\part1\12008.mp4'
    assert os.path.exists(vid_path)
    vid_name = os.path.basename(vid_path)
    anno = vid_annos[vid_name]

    out_path = r'D:\AI_data\expression_transfer\face1\output_00001.mp4'
    render_video(anno, out_path)


if __name__=="__main__":
    main()



