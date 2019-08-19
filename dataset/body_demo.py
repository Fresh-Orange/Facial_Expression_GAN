#-*- coding:utf-8 -*-

from __future__ import print_function

import os, sys
import cv2
import json
from tqdm import tqdm

# the order of  keypoints: 
# nose, neck
# right shoulder, right elbow, right wrist, left shoulder, left elbow, left wrist
# right hip, right knee, right ankle, left hip, left knee, left ankle
# right eye, left eye, right ear, left ear
body_keypoint_links = [
    [0,1], [0,14], [0,15], [1,2], [1,5], [1,8], [1,11],
    [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [11,12],
    [12,13], [14,16], [15,17]
]

colors = [
  (255.0, 0.0, 0.0),
  (255.0, 85.0, 0.0),
  (255.0, 170.0, 0.0),
  (255.0, 255.0, 0.0),
  (170.0, 255.0, 0.0),
  (85.0, 255.0, 0.0),
  (0.0, 255.0, 0.0),
  (0.0, 255.0, 85.0),
  (0.0, 255.0, 170.0),
  (0.0, 255.0, 255.0),
  (0.0, 170.0, 255.0),
  (0.0, 85.0, 255.0),
  (0.0, 0.0, 255.0),
  (255.0, 0.0, 255.0),
  (255.0, 0.0, 170.0), 
  (255.0, 0.0, 85.0), 
  (170.0, 0.0, 255.0)
]

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
            vid_anns[name][idx] = keypoints

    return vid_anns


def draw_keypoints(img, keypoint):
    flags = list()
    points = list()

    # draw points
    for i in range(0, len(keypoint), 3):
        x, y, flag = [int(k) for k in keypoint[i: i+3]]
        flags.append( flag )
        points.append( [x, y] )
        if flag == 0:      # keypoint not exist
            continue
        elif flag == 1:    # keypoint exist but invisible
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        elif flag == 2:    # keypoint exist and visible
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        else:
            raise ValueError("flag of keypoint must be 0, 1, or 2.")

    # draw links
    for (start, end), color in zip(body_keypoint_links, colors):
        if flags[start] == 0 or flags[end] == 0:
            continue
        else:
            x1, y1 = points[start]
            x2, y2 = points[end]
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


class VideoReader(object):
    def __init__(self, vid_path):
        self.vid = cv2.VideoCapture(vid_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    def read(self):
        while self.vid.isOpened():
            ret, frm = self.vid.read()
            if not ret:
                break
            yield frm
        self.vid.release()


def render_video(anno, out):
    vid_path = anno['path']
    vid = VideoReader(vid_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (vid.width, vid.height)
    vid_writer = cv2.VideoWriter(out, fourcc, vid.fps, size)

    for idx, frm in tqdm(enumerate(vid.read())):
        keypoint = anno[idx+1]
        draw_frm = draw_keypoints(frm, keypoint)
        vid_writer.write(draw_frm)
    vid_writer.release()


def main():
    json_path = "body_keypoint_train.json"
    vid_annos = load_json(json_path)

    vid_path = 'body/train/00001.mp4'
    assert os.path.exists(vid_path)
    vid_name = os.path.basename(vid_path)
    anno = vid_annos[vid_name]

    out_path = 'output_00001.mp4'
    render_video(anno, out_path)


if __name__=="__main__":
    main()



