import os
import argparse
from torch.backends import cudnn
import torch
from model.model import ExpressionGenerater as ExpressionGenerater
from model.model import NoTrackExpressionGenerater
import sys
from PIL import Image
from torchvision import transforms as T
import json
import numpy as np
import cv2
from tqdm import tqdm
import time
import math
from fusion.bg_move_fusion import knn_fusion
from utils.my_color_transfer import color_transfer

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
        self.sf_version = config.sf_version
        self.point_threshold = config.point_threshold

        self.G = ExpressionGenerater()
        self.sfG = ExpressionGenerater()
        self.colorG = NoTrackExpressionGenerater()





        #######   载入预训练网络   ######
        self.load()

        self.G.to(device)
        self.colorG.to(device)
        if config.eval == "1":
            self.G.eval()
        self.colorG.eval()


        self.transform = []
        self.transform.append(T.Resize((224, 224)))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(self.transform)

        self.vid_annos = self.load_json(json_file)

    def load(self):
        resume_iter = config.resume_iter
        color_resume_iter = config.color_resume_iter
        ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(self.version)
        sf_ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(self.sf_version)
        color_ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(self.color_version)
        if os.path.exists(os.path.join(ckpt_dir,
                                       '{}-G.ckpt'.format(resume_iter))):
            G_path = os.path.join(ckpt_dir,
                                  '{}-G.ckpt'.format(resume_iter))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            G_path = os.path.join(sf_ckpt_dir,
                                  '{}-G.ckpt'.format(121000))
            self.sfG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            cG_path = os.path.join(color_ckpt_dir,
                                   '{}-G.ckpt'.format(color_resume_iter))
            self.colorG.load_state_dict(torch.load(cG_path, map_location=lambda storage, loc: storage))
            print("load ckpt")
        else:
            print("found no ckpt")
            return None

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


    def draw_rotate_keypoints(self, img, bbox, keypoint, first_bbox, first_keypoint):
        flags = list()
        points = list()
        first_flags = list()
        first_points = list()
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

        fbx, fby, fbw, fbh = [int(v) for v in first_bbox]

        for i in range(0, len(first_keypoint), 3):
            x, y, flag = [int(k) for k in first_keypoint[i: i + 3]]
            x = int((x - fbx) / fbw * 224)
            y = int((y - fby) / fbh * 224)
            first_flags.append(flag)
            first_points.append([x, y])

        # 脸部对齐
        # 52是左眼的最左边， 61是右眼的最右边, x是宽度方向，y是高度方向
        if not (flags[52] == 0 or flags[61] == 0):
            left_x, left_y = points[52]
            right_x, right_y = points[61]
            if left_x > 224 / 4 or right_x < 224 - 224 / 4:
                return np.asarray(points_image), 0
            deltaH = right_y - left_y
            deltaW = right_x - left_x
            if math.sqrt(deltaW**2 + deltaH**2) < 1:
                return np.asarray(points_image), 0
            angle = math.asin(deltaH / math.sqrt(deltaW**2 + deltaH**2))
            angle = angle / math.pi * 180  # 弧度转角度

            # 计算第一帧的角度
            first_angle = 0
            if not (first_flags[52] == 0 or first_flags[61] == 0):
                left_x, left_y = first_points[52]
                right_x, right_y = first_points[61]
                deltaH = right_y - left_y
                deltaW = right_x - left_x
                if not math.sqrt(deltaW ** 2 + deltaH ** 2) < 1:
                    first_angle = math.asin(deltaH / math.sqrt(deltaW ** 2 + deltaH ** 2))
                    first_angle = first_angle / math.pi * 180  # 弧度转角度

            #print("angle", angle)
            #print("first_angle", first_angle)

            angle = angle - first_angle

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
            elif flag == 1 or flag == 2: # keypoint exist and visible
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



    def generate(self, first_frm_file, bg_file, alpha_file, body_file, first_frm_id, special_vids, hard2middle):
        first_frm = Image.open(first_frm_file)
        bg_body = np.array(first_frm)
        background = np.asarray(Image.open(bg_file))
        # print("shape", background.shape)
        body = np.asarray(Image.open(body_file))
        body_alpha = np.load(alpha_file)
        ## print("shape", body.shape)

        vid_name = os.path.basename(first_frm_file)
        vid_name = "{}.mp4".format(vid_name.split(".")[0])
        anno = self.vid_annos[vid_name]
        first_bbox, first_keypoint = anno[1]

        # 创建结果文件夹
        if not os.path.isdir("test_result"):
            os.mkdir("test_result")
        sample_dir = "test_result/gan-sample-{}-{}".format(self.version, self.config.resume_iter)
        if not os.path.isdir(sample_dir):
            os.mkdir(sample_dir)

        # 设置视频格式等
        frm = cv2.imread(first_frm_file)
        heigth, width, channels = frm.shape
        size = (width, heigth)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(sample_dir, '{}.mp4'.format(first_frm_id))
        vid_writer = cv2.VideoWriter(out_path, fourcc, 25, size)

        first_face_crop = None
        is_first = True
        fake_frm = None
        angle = None
        G = None

        # 特殊视频处理
        if first_frm_id.startswith("12") and int(first_frm_id) in special_vids:
            print("special {}-->{}".format(first_frm_id, hard2middle[first_frm_id]))
            vid_name = "{}.mp4".format(hard2middle[first_frm_id])
            temp_anno = self.vid_annos[vid_name]
            level2_first_bbox, level2_first_keypoint = temp_anno[1]
            level2_first_frm = os.path.join(first_frm_file[:-9], "{}.jpg".format(int(first_frm_id) - 1000))
            level2_first_frm = Image.open(level2_first_frm)
            _, first_face_crop = self.draw_bbox_keypoints(np.asarray(level2_first_frm), level2_first_bbox, level2_first_keypoint)


        match_cnt = 0
        rotate_match_cnt = 0
        # #########################
        # 主体循环，生成视频
        # #########################
        for idx in tqdm(range(len(anno) - 1)):
            #if idx % 4 == 0:
            #    self.load()
            bbox, keypoint = anno[idx + 1]
            if len(bbox) >= 2:
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])

            if is_first:
                is_first = False
                _, real_first_face_crop = self.draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
                if first_face_crop is None:
                    _, first_face_crop = self.draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)
                if len(bbox) == 0:
                    #print("no first bbox")
                    vid_writer.write(frm)
                    is_first = True # 重置first
                    continue
                if len(first_bbox) == 0:
                    first_bbox = bbox
                    first_keypoint = keypoint
                    #print("first_bbox = bbox")
                vid_writer.write(cv2.cvtColor(np.asarray(first_frm), cv2.COLOR_RGB2BGR))
                continue
            else:
                # 获取第idx帧的关键点图
                draw_kp, angle = self.draw_rotate_keypoints(np.asarray(first_frm), bbox, keypoint, first_bbox, first_keypoint)
                draw_kp_noratota, _ = self.draw_bbox_keypoints(np.asarray(first_frm), bbox, keypoint)

                #print(angle)
                if draw_kp is None or first_face_crop is None:
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

            first_draw_kp, _ = self.draw_bbox_keypoints(np.asarray(first_frm), first_bbox, first_keypoint)
            # print(np.mean(first_draw_kp - draw_kp_noratota))

            #print(angle)
            if angle == 360:
                angle = 0
                G = self.sfG
            else:
                G = self.G

            # 特殊帧特殊处理，若目标帧与原始帧很接近，那么直接拿原始帧
            point_threshold = self.point_threshold
            if np.mean(first_draw_kp - draw_kp_noratota) < point_threshold:
                match_cnt += 1
                ##print(np.mean(first_draw_kp - draw_kp_noratota))
                #print("match ")
                fake_frm = knn_fusion(bg_body, background, body, body_alpha, first_bbox, np.asarray(first_face_crop), bbox)
                fake_frm = cv2.cvtColor(np.asarray(fake_frm), cv2.COLOR_RGB2BGR)
                fake_frm = cv2.resize(fake_frm, (width, heigth))
                vid_writer.write(fake_frm)
                continue

            # 调用模型生成一帧
            def generate_frm(draw_kp, first_face_tensor):
                img_kp = Image.fromarray(draw_kp, 'RGB')
                # img.show()
                key_points = self.transform(img_kp)
                key_points = key_points.unsqueeze(0)
                first_face_tensor = first_face_tensor.to(device)
                key_points = key_points.to(device)
                face_fake = G(first_face_tensor, key_points)
                return face_fake

            def to_PIL(face_fake):
                frm = denorm(face_fake.data.cpu())
                frm = frm[0]
                toPIL = T.ToPILImage()
                frm = toPIL(frm)
                return frm

            # 特殊帧特殊处理，若目标帧与原始帧很接近，那么直接拿原始帧
            if np.mean(first_draw_kp - draw_kp) < point_threshold:
                rotate_match_cnt += 1
                #print(np.mean(first_draw_kp - draw_kp))
                frm = Image.fromarray(np.array(first_face_crop))
                frm = frm.resize(size=(224, 224))
                #print("match rotation")
            else:
                face_fake = generate_frm(draw_kp, first_face_tensor)
                frm = to_PIL(face_fake)



            # 如果有旋转
            if angle != 0:
                face_fake_norotate = generate_frm(draw_kp_noratota, first_face_tensor)
                frm = frm.rotate(-angle)
                norotate_frm = to_PIL(face_fake_norotate)

                # 旋转的人脸和未旋转人脸的融合
                frm = np.array(frm)
                frm.flags.writeable = True
                norotate_frm = np.asarray(norotate_frm)
                mask = np.mean(frm, axis=2)
                frm[:, :, 0] = np.where(mask < 0.1, norotate_frm[:, :, 0], frm[:, :, 0])
                frm[:, :, 1] = np.where(mask < 0.1, norotate_frm[:, :, 1], frm[:, :, 1])
                frm[:, :, 2] = np.where(mask < 0.1, norotate_frm[:, :, 2], frm[:, :, 2])
                frm = Image.fromarray(frm)

            #save_image(frm, "test_result/fake_face_{}.jpg".format(idx))

            if config.color_transfer == "1":
                fake_face = cv2.cvtColor(np.asarray(frm), cv2.COLOR_RGB2BGR)
                first_face = cv2.cvtColor(np.asarray(Image.fromarray(first_face_crop, 'RGB')), cv2.COLOR_RGB2BGR)
                frm = color_transfer(first_face, fake_face)
                frm = cv2.cvtColor(np.asarray(frm), cv2.COLOR_BGR2RGB)



            if config.sharpen == "1":
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) * config.sharpen_lambda
                kernel[1,1] = kernel[1,1] + 1
                frm = cv2.filter2D(frm, -1, kernel)

            ###### 融合
            #fake_frm = fusion(first_img, first_bbox, np.asarray(frm), bbox)
            fake_frm = knn_fusion(bg_body, background, body, body_alpha, first_bbox, np.asarray(frm), bbox)
            #
            fake_frm = cv2.cvtColor(np.asarray(fake_frm), cv2.COLOR_RGB2BGR)
            #cv2.imwrite("test_result/fake_frm_{}.jpg".format(idx), fake_frm)
            fake_frm = cv2.resize(fake_frm, (width, heigth))

            # cv2.imwrite("test_result/fake_frm_{}.jpg".format(idx), fake_frm)

            vid_writer.write(fake_frm)
        vid_writer.release()
        print("match_cnt {}, rotate_match_cnt {}".format(match_cnt, rotate_match_cnt))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--resume_iter', type=int, default=11000)
    parser.add_argument('--color_resume_iter', type=int, default=3000)
    parser.add_argument('--version', type=str, default="256-level2")
    parser.add_argument('--sf_version', type=str, default="256-level123_ResD_ResID_sideface_finetune123_fasttrack")
    parser.add_argument('--color_version', type=str, default="256-level2_color-transfer")
    parser.add_argument('--gpu', type=str, default="2")
    parser.add_argument('--test_level', type=int, default=1)
    parser.add_argument('--color_transfer', type=str, default="1")
    parser.add_argument('--sharpen', type=str, default="0")             # 锐化效果不好，因此默认为0
    parser.add_argument('--sharpen_lambda', type=float, default=0.4)
    parser.add_argument('--eval', type=str, default="0")
    parser.add_argument('--specific', type=str, default="")
    parser.add_argument('--specific_list', '--list', nargs='+')
    parser.add_argument('--point_threshold', type=float, default=1.9)


    config = parser.parse_args()

    VG = VideoGenerator(config, "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/face_keypoint_test.json")

    test_file = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/test_first_frame/{}.jpg"

    test_dir = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/test_first_frame"

    bg_file = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/background/knn_reseg_{}__bg.png"

    body_file = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/body/{}_.png"

    alpha_file = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/body/{}_alpha.npy"

    hard2middle = {'12046': '11046',
                   '12084': '11085'}

    for f in sorted(os.listdir(test_dir)):
        id = f.split(".")[0]
        match = config.specific if config.specific != "" else str(10+config.test_level-1)
        if id.startswith(match) and int(id) > 11050 and int(id) < 11104:
            print(id)
            VG.generate(test_file.format(id), bg_file.format(id), alpha_file.format(id), body_file.format(id), id,
                        special_vids=[12046], hard2middle=hard2middle)
            time.sleep(1)