import cv2
import os

test_dir = "test_first_frm"
border_dir = "border_first_frm"

for f in os.listdir(test_dir):
    print(f)
    test_file = os.path.join(test_dir, f)
    border_file = os.path.join(border_dir, f)
    test_img = cv2.imread(test_file)
    border_img = cv2.imread(border_file)
    ih, iw, c = test_img.shape
    border_w = int(iw/5) if int(iw/5) % 2 == 0 else int(iw/5)+1
    border_h = int(ih/7) if int(ih/7) % 2 == 0 else int(ih/7)+1
    border_img = cv2.resize(border_img, (int(iw+border_w), int(ih+border_h)))
    bh, bw, c = border_img.shape
    # 计算padding大小
    padding_w = (bw - iw) // 2
    padding_h = (bh - ih) // 2
    border500 = border_img.copy()
    border500[padding_h:-padding_h, padding_w:-padding_w] = test_img
    # step = 50
    # step_num = 4
    # for i in range(step_num):
    #     border500 = cv2.copyMakeBorder(border500, step, step, step, step,
    #                                    borderType=cv2.BORDER_REFLECT101)
    border500 = cv2.copyMakeBorder(border500, 500-padding_h, 500-padding_h, 500-padding_w, 500-padding_w, borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite("border_const500/{}".format(f), border500)
