import cv2
import numpy

def fusion(knockout_bg, bg_bbox, face, face_bbox):
    bg_bx, bg_by, bg_bw, bg_bh = [int(v) for v in bg_bbox]
    f_bx, f_by, f_bw, f_bh = [int(v) for v in face_bbox]

    white_bord = numpy.ones_like(knockout_bg)

    white_bord = cv2.copyMakeBorder(white_bord, 500, 500, 500, 500, cv2.BORDER_REPLICATE)

    # 压缩放大
    ratio_w = f_bw / bg_bw  # 宽度压缩比
    ratio_h = f_bh / bg_bh  # 长度压缩比
    heigth, width , _ = knockout_bg.shape
    knockout_bg = cv2.resize(knockout_bg, (int(width*ratio_w), int(heigth*ratio_h)))

    no_pad_h, no_pad_w , _ = knockout_bg.shape

    # 这里的4 到 -4 有待商榷
    knockout_bg = cv2.copyMakeBorder(knockout_bg[4:-4, 4:-4], 1004, 1004, 1004, 1004, cv2.BORDER_REPLICATE)  # 背景的边缘扩展

    cv2.imwrite("knockout_bg.jpg", knockout_bg)

    relative_bg_bbox = [bg_bx / width * no_pad_w + 1000, bg_by / heigth * no_pad_h + 1000, knockout_bg.shape[1], knockout_bg.shape[0]]

    face = cv2.resize(face, (f_bw, f_bh))


    # 画背景
    bg_bx, bg_by, bg_bw, bg_bh = [int(v) for v in relative_bg_bbox]
    left = bg_bx - f_bx
    right = left + width
    up = bg_by - f_by
    bottom = up + heigth

    print("corner", left, right, up, bottom)


    white_bord[500:-500,500:-500] = knockout_bg[up:bottom, left:right]

    # 画人脸
    white_bord[f_by+500+1:f_by+f_bh+500+1, f_bx+500+1:f_bx+f_bw+500+1] = face[:, :]

    white_bord = white_bord[500:-500,500:-500]

    return white_bord

def stepBorderV2(src, top, bottom, left, right, step):
    step_num = top // step + 1
    padding = src
    padding = cv2.copyMakeBorder(padding, top, bottom, left, right,
                                 borderType=cv2.BORDER_REPLICATE)
    top_slice = src[:step]
    top_block = numpy.tile(top_slice, (step_num, 1, 1))
    padding[:top] = top_block[:top]

    step_num = bottom // step + 1
    bottom_slice = src[-step:]
    bottom_block = numpy.tile(bottom_slice, (step_num, 1, 1))
    padding[-bottom:] = bottom_block[-bottom:]

    step_num = left // step + 1
    left_slice = src[:, :left]
    left_block = numpy.tile(left_slice, (1, step_num, 1))
    padding[:, :left] = left_block[:, :left]

    step_num = right // step + 1
    right_slice = src[:, -right:]
    right_block = numpy.tile(right_slice, (step_num, 1, 1))
    padding[:, -right:] = right_block[:, -right:]

    return padding

def stepBorder(src, top, bottom, left, right, step):
    step_num = top // step
    padding = src
    for i in range(step_num):
        padding = cv2.copyMakeBorder(padding, step, 0, 0, 0,
                                       borderType=cv2.BORDER_REFLECT101)
    padding = cv2.copyMakeBorder(padding, top % step, 0, 0, 0,
                                 borderType=cv2.BORDER_REFLECT101)

    step_num = bottom // step
    for i in range(step_num):
        padding = cv2.copyMakeBorder(padding, 0, step, 0, 0,
                                     borderType=cv2.BORDER_REFLECT101)
    padding = cv2.copyMakeBorder(padding, 0, bottom % step, 0, 0,
                                 borderType=cv2.BORDER_REFLECT101)

    step_num = left // step
    for i in range(step_num):
        padding = cv2.copyMakeBorder(padding, 0, 0, step, 0,
                                     borderType=cv2.BORDER_REFLECT101)
    padding = cv2.copyMakeBorder(padding, 0, 0, left % step, 0,
                                 borderType=cv2.BORDER_REFLECT101)

    step_num = right // step
    for i in range(step_num):
        padding = cv2.copyMakeBorder(padding, 0, 0, 0, step,
                                     borderType=cv2.BORDER_REFLECT101)
    padding = cv2.copyMakeBorder(padding, 0, 0, 0, left % step,
                                 borderType=cv2.BORDER_REFLECT101)

    return padding


def knn_fusion(bg, body, body_alpha, bg_bbox, face, face_bbox):
    bg_bx, bg_by, bg_bw, bg_bh = [int(v) for v in bg_bbox]
    f_bx, f_by, f_bw, f_bh = [int(v) for v in face_bbox]
    # print("bg_bbox", bg_bbox)
    # print("face_bbox", face_bbox)
    heigth, width, _ = body.shape
    middle_shift = int(bg_bx + bg_bw/2 - width/2) # 人脸的中心偏离图像中心的大小

    white_bord = numpy.ones_like(bg)

    white_bord = cv2.copyMakeBorder(white_bord, 500, 500, 500, 500, cv2.BORDER_REPLICATE)

    # 压缩放大
    ratio_w = f_bw / bg_bw  # 宽度压缩比
    ratio_h = f_bh / bg_bh  # 长度压缩比
    heigth, width , _ = body.shape
    # print("new shape", body.shape)
    body = cv2.resize(body, (int(width*ratio_w), int(heigth*ratio_h)))
    body_alpha = cv2.resize(body_alpha, (int(width*ratio_w), int(heigth*ratio_h)))

    no_pad_h, no_pad_w , _ = body.shape

    #body = cv2.copyMakeBorder(body, 1000, 1000, 1000, 1000, cv2.BORDER_REPLICATE)  # 身体向下填充 pad only bottom
    step = 5
    left_padding = max(bg_bw // 5 - middle_shift, 0)
    right_padding = max(bg_bw // 5 + middle_shift, 0)
    body = stepBorderV2(body, 0, 1000, 0, 0, step)  # 身体向下填充 pad only bottom
    body = stepBorderV2(body, 0, 0, left_padding, right_padding, step)  # 身体向下填充 pad only bottom
    body = cv2.copyMakeBorder(body, 1000, 0, 1000-(left_padding), 1000-(right_padding), value=0, borderType=cv2.BORDER_CONSTANT)  # 身体向下填充 pad only bottom

    body_alpha = stepBorderV2(body_alpha, 0, 1000, 0, 0, step)  # 身体向下填充 pad only bottom
    body_alpha = stepBorderV2(body_alpha, 0, 0, left_padding, right_padding, step)  # 身体向下填充 pad only bottom
    body_alpha = cv2.copyMakeBorder(body_alpha, 1000, 0, 1000 - (left_padding), 1000 - (right_padding), value=0,
                              borderType=cv2.BORDER_CONSTANT)  # 身体向下填充 pad only bottom

    #cv2.imwrite("knockout_bg.jpg", knockout_bg)

    relative_bg_bbox = [bg_bx / width * no_pad_w + 1000, bg_by / heigth * no_pad_h + 1000, body.shape[1], body.shape[0]]

    face = cv2.resize(face, (f_bw, f_bh))

    # 画背景
    white_bord[500:-500, 500:-500] = bg[:, :]

    # 画人体
    bg_bx, bg_by, bg_bw, bg_bh = [int(v) for v in relative_bg_bbox]
    left = bg_bx - f_bx
    right = left + width
    up = bg_by - f_by
    bottom = up + heigth

    # print("corner", left, right, up, bottom)
    mask = numpy.mean(body[up:bottom, left:right, :], axis=2)
    # white_bord[500:-500, 500:-500, 0] = numpy.where(mask < 0.1, white_bord[500:-500,500:-500, 0], body[up:bottom, left:right, 0])
    # white_bord[500:-500, 500:-500, 1] = numpy.where(mask < 0.1, white_bord[500:-500,500:-500, 1], body[up:bottom, left:right, 1])
    # white_bord[500:-500, 500:-500, 2] = numpy.where(mask < 0.1, white_bord[500:-500,500:-500, 2], body[up:bottom, left:right, 2])

    white_bord[500:-500, 500:-500, 0] = (1-body_alpha[up:bottom, left:right])*white_bord[500:-500, 500:-500, 0] + \
                                        body_alpha[up:bottom, left:right]*body[up:bottom, left:right, 0]
    white_bord[500:-500, 500:-500, 1] = (1-body_alpha[up:bottom, left:right])*white_bord[500:-500, 500:-500, 1] + \
                                        body_alpha[up:bottom, left:right]*body[up:bottom, left:right, 1]
    white_bord[500:-500, 500:-500, 2] = (1-body_alpha[up:bottom, left:right])*white_bord[500:-500, 500:-500, 2] + \
                                        body_alpha[up:bottom, left:right]*body[up:bottom, left:right, 2]


    # 画人脸
    white_bord[f_by+500+1:f_by+f_bh+500+1, f_bx+500+1:f_bx+f_bw+500+1] = face[:, :]

    white_bord = white_bord[500:-500,500:-500]

    return white_bord


if __name__ == '__main__':
    pass
