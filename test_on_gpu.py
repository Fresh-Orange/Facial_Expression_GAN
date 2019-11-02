import os
import argparse
from torch.backends import cudnn
import torch
from model.model import FusionGenerater
from model.model import TwoPointBigConvExpressionGenerater as ExpressionGenerater
import sys
from PIL import Image
from torchvision import transforms as T
import json
import numpy as np
import cv2
from tqdm import tqdm
import time
from fusion.bg_move_fusion import fusion


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

class VideoGenerator():
    def __init__(self, config, json_file):
        # For fast training.
        self.config = config
        cudnn.benchmark = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if len(sys.argv) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        global device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.version = config.version
        self.color_version = config.color_version

        self.G = ExpressionGenerater()
        self.colorG = NoTrackExpressionGenerater()

        ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/fusion-ckpt-{}".format(config.fusion_version)
        self.FG = FusionGenerater()
        FG_path = os.path.join(ckpt_dir,
                               '{}-G.ckpt'.format(config.fusion_resume_iter))
        self.FG.load_state_dict(torch.load(FG_path, map_location=lambda storage, loc: storage))
        self.FG.to(device)
        self.FG.eval()

        #######   载入预训练网络   ######
        resume_iter = config.resume_iter
        color_resume_iter = config.color_resume_iter
        ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(self.version)
        color_ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(self.color_version)
        if os.path.exists(os.path.join(ckpt_dir,
                                       '{}-G.ckpt'.format(resume_iter))):
            G_path = os.path.join(ckpt_dir,
                                  '{}-G.ckpt'.format(resume_iter))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            cG_path = os.path.join(color_ckpt_dir,
                                  '{}-G.ckpt'.format(color_resume_iter))
            self.colorG.load_state_dict(torch.load(cG_path, map_location=lambda storage, loc: storage))
            print("load ckpt")
        else:
            print("found no ckpt")
            return None

        self.G.to(device)
        self.G.eval()


        self.transform = []
        self.transform.append(T.Resize((224, 224)))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(self.transform)

        self.vid_annos = self.load_json(json_file)

    #############  process data  ##########
    def crop_face(self, img, bbox, keypoint):
        flags = list()
        points = list()
        # can not detect face in some images
        if len(bbox) == 0:
            return None
        # draw bbox
        x, y, w, h = [int(v) for v in bbox]
        crop_img = img[y:y + h, x:x + w]
        return crop_img

    def load_json(self, path):
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


    def draw_rotate_keypoints(self, img, bbox, keypoint):
        flags = list()
        points = list()
        # can not detect face in some images
        if len(bbox) == 0:
            return None, None
        points_image = np.zeros((224, 224, 3), np.uint8)
        # draw bbox
        bx, by, bw, bh = [int(v) for v in bbox]


        for i in range(0, len(keypoint), 3):
            x, y, flag = [int(k) for k in keypoint[i: i + 3]]
            x = int((x - bx) / bw * 224)
            y = int((y - by) / bh * 224)
            flags.append(flag)
            points.append([x, y])
            if flag == 0:  # keypoint not exist
                continue
            elif flag == 1 or flag == 2: # keypoint exist and visible
                cv2.circle(points_image, (x, y), 2, (0, 255, 0), -1)
            else:
                raise ValueError("flag of keypoint must be 0, 1, or 2.")

        # 脸部对齐
        # 52是左眼的最左边， 61是右眼的最右边, x是宽度方向，y是高度方向
        if not (flags[52] == 0 or flags[61] == 0):
            left_x, left_y = points[52]
            right_x, right_y = points[61]
            deltaH = right_y - left_y
            deltaW = right_x - left_x
            if math.sqrt(deltaW**2 + deltaH**2) < 1:
                return np.asarray(points_image), 0
            angle = math.asin(deltaH / math.sqrt(deltaW**2 + deltaH**2))
            angle = angle / math.pi * 180  # 弧度转角度

            if abs(angle) < 5:
                return np.asarray(points_image), 0
            points_image = Image.fromarray(np.uint8(points_image))
            points_image = points_image.rotate(angle)
        else:
            angle = 0

        return np.asarray(points_image), angle


    def draw_bbox_keypoints(self, img, bbox, keypoint):
        flags = list()
        points = list()
        # can not detect face in some images
        if len(bbox) == 0:
            return None, None
        points_image = np.zeros((224, 224, 3), np.uint8)
        # draw bbox
        bx, by, bw, bh = [int(v) for v in bbox]


        for i in range(0, len(keypoint), 3):
            x, y, flag = [int(k) for k in keypoint[i: i + 3]]
            x = int((x - bx) / bw * 224)
            y = int((y - by) / bh * 224)
            flags.append(flag)
            points.append([x, y])
            if flag == 0:  # keypoint not exist
                continue
            elif flag == 1:  # keypoint exist but invisible
                cv2.circle(points_image, (x, y), 2, (0, 0, 255), -1)
            elif flag == 2:  # keypoint exist and visible
                cv2.circle(points_image, (x, y), 2, (0, 255, 0), -1)
            else:
                raise ValueError("flag of keypoint must be 0, 1, or 2.")

        return points_image, self.crop_face(img, bbox, keypoint)



    def extract_image(self, img, bbox):
        # can not detect face in some images
        Knockout_image = img.copy()
        # draw bbox
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(Knockout_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
        return Knockout_image, img



    def generate(self, first_frm_file, first_frm_id):
        first_frm = Image.open(first_frm_file)

        vid_name = os.path.basename(first_frm_file)
        vid_name = "{}.mp4".format(vid_name.split(".")[0])
        anno = self.vid_annos[vid_name]
        first_bbox, first_keypoint = anno[1]

        _, first_knockout_image, _, first_img, _ = self.extract_image(np.array(first_frm), first_bbox, first_keypoint)

        frm = cv2.imread(first_frm_file)
        heigth, width, channels = frm.shape
        # print(heigth, width)
        size = (width, heigth)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.isdir("test_result"):
            os.mkdir("test_result")
        sample_dir = "test_result/gan-sample-{}-{}".format(self.version, self.config.resume_iter)
        if not os.path.isdir(sample_dir):
            os.mkdir(sample_dir)
        out_path = os.path.join(sample_dir, '{}.mp4'.format(first_frm_id))
        vid_writer = cv2.VideoWriter(out_path, fourcc, 25, size)

        first_face_crop = None
        is_first = True
        fake_frm = None
        first_kp_img = None
        for idx in tqdm(range(len(anno) - 1)):
            # print(idx)
            # if (idx+1) % 5 != 1:
            #     continue

            bbox, keypoint = anno[idx + 1]
            if len(bbox) >= 2:
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])

            if is_first:
                is_first = False
                draw_frm, frm_crop = self.draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
                first_kp_img = draw_frm.copy()
                if draw_frm is None:
                    print("1 no bbox")
                    vid_writer.write(frm)
                    continue
                frm_crop_arr = Image.fromarray(frm_crop, 'RGB')
                # frm_crop_arr.save("test_result/first.jpg")
                first_frm_tensor = self.transform(frm_crop_arr)
                first_frm_tensor = first_frm_tensor.unsqueeze(0)
            else:
                draw_frm, _ = self.draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
                if draw_frm is None or frm_crop is None:
                    print("no bbox")
                    if first_face_crop is None:
                        vid_writer.write(frm)
                        continue
                    if fake_frm is None:
                        vid_writer.write(frm)
                    else:
                        vid_writer.write(fake_frm)
                    continue

                first_face_crop_arr = Image.fromarray(first_face_crop, 'RGB')
                first_face_tensor = self.transform(first_face_crop_arr)
                first_face_tensor = first_face_tensor.unsqueeze(0)

            first_kp = Image.fromarray(first_kp_img, 'RGB')
            img = Image.fromarray(draw_frm, 'RGB')
            # img.show()
            key_points = self.transform(img)
            key_points = key_points.unsqueeze(0)
            first_kp = self.transform(first_kp)
            first_kp = first_kp.unsqueeze(0)
            first_frm_tensor = first_frm_tensor.to(device)
            key_points = key_points.to(device)
            first_kp = first_kp.to(device)
            face_fake = self.G(first_frm_tensor, first_kp, key_points)

            frm = denorm(face_fake.data.cpu())

            toPIL = T.ToPILImage()
            frm = toPIL(frm.squeeze())
            #
            save_frm = cv2.cvtColor(np.asarray(frm), cv2.COLOR_RGB2BGR)
            #cv2.imwrite("test_result/fake_face_{}.jpg".format(idx), save_frm)

            ###### 融合
            fake_frm = fusion(first_knockout_image, first_bbox, np.asarray(frm), bbox)
            #
            fake_frm = cv2.cvtColor(np.asarray(fake_frm), cv2.COLOR_RGB2BGR)
            #cv2.imwrite("test_result/fake_frm_{}.jpg".format(idx), fake_frm)
            fake_frm = cv2.resize(fake_frm, (width, heigth))

            # cv2.imwrite("test_result/fake_frm_{}.jpg".format(idx), fake_frm)

            vid_writer.write(fake_frm)
        vid_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--resume_iter', type=int, default=11000)
    parser.add_argument('--color_resume_iter', type=int, default=3000)
    parser.add_argument('--version', type=str, default="256-level2")
    parser.add_argument('--color_version', type=str, default="256-level2_color-transfer")
    parser.add_argument('--gpu', type=str, default="2")
    parser.add_argument('--test_level', type=int, default=1)


    config = parser.parse_args()

    VG = VideoGenerator(config, "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/face_keypoint_test.json")

    test_file = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/test_first_frame/{}.jpg"

    test_dir = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/test_first_frame"

    for f in sorted(os.listdir(test_dir)):
        id = f.split(".")[0]
        if id.startswith("1104"):#str(10+config.test_level-1)
            print(id)
            VG.generate(test_file.format(id), id)
            time.sleep(1)