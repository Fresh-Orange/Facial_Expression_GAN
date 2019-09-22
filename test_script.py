import os
import time

cmd = "python3 test_on_gpu.py --resume_iter 100000 --gpu 4 --version 256-level2 --test_image {}"

test_dir = "/media/data2/laixc/AI_DATA/expression_transfer/face_test_dataset/face/test_first_frame"

for f in sorted(os.listdir(test_dir)):
    id = f.split(".")[0]
    if id.startswith("12"):
        full_cmd = cmd.format(id)
        print("cmd : ", full_cmd)
        os.system(full_cmd)
        time.sleep(2)
