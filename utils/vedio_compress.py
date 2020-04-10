import os
import time

input_dir = "/media/data2/laixc/Facial_Expression_GAN/test_result/" \
            "gan-sample-256-level2_ResD_ResID_finetune_aug_fasttrack-110000-level2-knockout-100000"

output_dir = "/media/data2/laixc/Facial_Expression_GAN/test_result/compress_level2_ResD_ResID_finetune2_aug_fasttrack-110000"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

crf = 18

for f in os.listdir(input_dir):
    full_path = os.path.join(input_dir, f)
    out_path = os.path.join(output_dir, f)
    cmd = "ffmpeg -i {} -vcodec libx264 -crf {} {}".format(full_path, crf, out_path)
    print(cmd)
    os.system(cmd)
    time.sleep(2)