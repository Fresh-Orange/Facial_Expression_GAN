import os
import argparse
from solver import Solver
import data_loader
from torch.backends import cudnn
import torch
from model import LandMarksDetect, RealFakeDiscriminator, ExpressionGenerater, FeatureExtractNet
from torchvision.utils import save_image
import math
import sys
from PIL import Image
from torchvision import transforms as T
import json
import numpy as np
import cv2


device = None

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def to_image(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    x = denorm(x.data.cpu())
    ndarr = x.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = ndarr
    return im


def generate_video(config, first_frm_file, json_file):
    # For fast training.
    cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(sys.argv) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    version = config.version

    G = ExpressionGenerater()

    #######   载入预训练网络   ######
    resume_iter = config.resume_iter
    ckpt_dir = r"C:\Users\FreshOrange\Desktop\tmp"
    if os.path.exists(os.path.join(ckpt_dir,
                                  '{}-G.ckpt'.format(resume_iter))):
        G_path = os.path.join(ckpt_dir,
                              '{}-G.ckpt'.format(resume_iter))
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    else:
        return None

    G.to(device)
    G.eval()

    #############  process data  ##########
    def crop_face(img, bbox, keypoint):
        flags = list()
        points = list()

        # can not detect face in some images
        if len(bbox) == 0:
            return None

        # draw bbox
        x, y, w, h = [int(v) for v in bbox]
        crop_img = img[y:y + h, x:x + w]
        return crop_img

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
            return None

        points_image = np.zeros_like(img, np.uint8)

        # draw bbox
        # x, y, w, h = [int(v) for v in bbox]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # draw points
        for i in range(0, len(keypoint), 3):
            x, y, flag = [int(k) for k in keypoint[i: i + 3]]
            flags.append(flag)
            points.append([x, y])
            if flag == 0:  # keypoint not exist
                continue
            elif flag == 1:  # keypoint exist but invisible
                cv2.circle(points_image, (x, y), 3, (0, 0, 255), -1)
            elif flag == 2:  # keypoint exist and visible
                cv2.circle(points_image, (x, y), 3, (0, 255, 0), -1)
            else:
                raise ValueError("flag of keypoint must be 0, 1, or 2.")

        return crop_face(points_image, bbox, keypoint), crop_face(img, bbox, keypoint)

    transform = []
    transform.append(T.Resize((224,224)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    first_frm = Image.open(first_frm_file)

    vid_annos = load_json(json_file)
    vid_name = os.path.basename(first_frm_file)
    vid_name = "{}.mp4".format(vid_name.split(".")[0])
    anno = vid_annos[vid_name]

    frm = cv2.imread(first_frm_file)
    _, _, channels = frm.shape
    size = (224, 224)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = r'C:\Users\FreshOrange\Desktop\tmp\out_level2_11088_{}.mp4'.format(config.resume_iter)
    vid_writer = cv2.VideoWriter(out_path, fourcc, 10, size)

    print(len(anno))
    frm_crop = None
    is_first = True
    for idx in range(len(anno)-1):
        print(idx)
        if (idx+1) % 3 == 0:
            continue

        bbox, keypoint = anno[idx+1]
        if is_first:
            is_first = False
            draw_frm, frm_crop = draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
            frm_crop_arr = Image.fromarray(frm_crop, 'RGB')
            first_frm_tensor = transform(frm_crop_arr)
            first_frm_tensor = first_frm_tensor.unsqueeze(0)
        else:
            draw_frm, _ = draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
            frm_crop_arr = Image.fromarray(frm_crop, 'RGB')
            first_frm_tensor = transform(frm_crop_arr)
            first_frm_tensor = first_frm_tensor.unsqueeze(0)

        img = Image.fromarray(draw_frm, 'RGB')
        # img.show()
        key_points = transform(img)
        key_points = key_points.unsqueeze(0)
        face_fake = G(first_frm_tensor, key_points)

        sample_path_rec = os.path.join(r"C:\Users\FreshOrange\Desktop\tmp", '{}-image-face_fake.jpg'.format(idx + 1))
        save_image(denorm(face_fake.data.cpu()), sample_path_rec)

        frm = denorm(face_fake.data.cpu())
        toPIL = T.ToPILImage()
        frm = toPIL(frm.squeeze())

        frm = cv2.cvtColor(np.asarray(frm), cv2.COLOR_RGB2BGR)

        vid_writer.write(frm)
    vid_writer.release()

    # faces_fake = G(faces, target_points)

    # TODO: 放回原视频


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--resume_iter', type=int, default=54000)
    parser.add_argument('--version', type=str, default="256-level2")
    parser.add_argument('--dataset_level', '--list', nargs='+',
                        default=['easy', 'middle', 'hard'])
    parser.add_argument('--gpu', type=str, default="2")


    config = parser.parse_args()

    print("gene")
    generate_video(config, r"D:\AI_data\expression_transfer\face_test_dataset\face\test_first_frame\11088.jpg"
                   , r"D:\AI_data\expression_transfer\face_test_dataset\face\face_keypoint_test.json")

