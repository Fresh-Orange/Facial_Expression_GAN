#-*- coding:utf-8 -*-

from __future__ import print_function

import os, sys
import cv2
import json
from tqdm import tqdm
from dataset.body_demo import VideoReader


# the order of  keypoints show in the sample_face.jpg
def get_face_landmark_links():
    landmark_links = list()
    landmark_links.extend( [[i, i+1] for i in range(32)] )        # 脸框

    landmark_links.extend( [[i, i+1] for i in range(33, 37)] )    # 左眉
    landmark_links.extend( [[i, i+1] for i in range(64, 67)] )    
    landmark_links.extend( [[33, 64], [37, 67]] )                 
    
    landmark_links.extend( [[i, i+1] for i in range(38, 42)] )    # 右眉
    landmark_links.extend( [[i, i+1] for i in range(68, 71)] )    
    landmark_links.extend( [[38, 68], [42, 71]] )                 
    
    landmark_links.extend( [[52,53], [53,72], [72,54], [54,55], 
                            [55,56], [56,73], [73,57], [57,52], 
                            [74,72], [74,73], [104,72], [104,73]] )   # 左眼眶

    landmark_links.extend( [[58,59], [59,75], [75,60], [60,61], 
                            [61,62], [62,76], [76,63], [63,58], 
                            [77,75], [77,76], [105,75], [105,76]] )   # 右眼眶

    landmark_links.extend( [[i, i+1] for i in range(43, 46)] )    # 鼻梁
    landmark_links.extend( [[43,78], [43,79], [46,48], [46,50]] )
    landmark_links.extend( [[78,80], [80,82], [82,47]] )
    landmark_links.extend( [[79,81], [81,83], [83,51]] )
    landmark_links.extend( [[i, i+1] for i in range(47, 51)] )

    landmark_links.extend( [[i, i+1] for i in range(84, 95)] )    # 嘴唇
    landmark_links.extend( [[i, i+1] for i in range(96, 103)] )
    landmark_links.extend( [[84,95], [84,96], [96,103], [90,100]] )

    return landmark_links


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

    # draw links
    # face_keypoint_links = get_face_landmark_links()
    # for start, end in face_keypoint_links:
    #     if flags[start] == 0 or flags[end] == 0:
    #         continue
    #     else:
    #         x1, y1 = points[start]
    #         x2, y2 = points[end]
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def render_video(first_frm, anno, out):
    frm = cv2.imread(first_frm)
    frm.copy()
    height, width, channels = frm.shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(out, fourcc, 30, size)

    print(len(anno))
    for idx in range(len(anno)-1):
        frm_copy = frm.copy()
        bbox, keypoint = anno[idx+1]
        draw_frm = draw_bbox_keypoints(frm_copy, bbox, keypoint)
        vid_writer.write(draw_frm)
    vid_writer.release()


def main():
    json_path = r"D:\AI_data\expression_transfer\face_test_dataset\face\face_keypoint_test.json"
    vid_annos = load_json(json_path)
    for vid in range(12097, 12098):
        vid_path = r'D:\AI_data\expression_transfer\face_test_dataset\face\test_first_frame\{}.jpg'.format(vid)
        assert os.path.exists(vid_path)
        vid_name = os.path.basename(vid_path)
        vid_name = "{}.mp4".format(vid_name.split(".")[0])
        anno = vid_annos[vid_name]

        out_path = r'D:\AI_data\expression_transfer\face_test_dataset\face\output_{}.mp4'.format(vid)
        render_video(vid_path, anno, out_path)


if __name__=="__main__":
    main()



