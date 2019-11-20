# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    # (100, 2)
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """

    Args:
        center: 原图的中心点坐标
        scale: 原图最大边长
        rot: 旋转角度
        output_size: 输出图片的大小
        shift:
        inv: 变换的方向

    Returns:

    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    # TODO: 感觉这里 src_w, dst_w 容易让人混淆, src_h, dst_h 似乎更好
    # 表示的以坐标原点为中心点的正方形的, (0, -h/2) 这个点
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # 不考虑 shift, 表示的是原图和输出图的中心点
    src[0, :] = center + scale_tmp * shift
    # 不考虑 shift, 表示的是原图和输出图的中心点正上方的点
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    # 左上方的点
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, src, dst


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """
    绕 (0, 0) 旋转 rot_rad, rot_rad > 0 表示顺时针旋转, rot_rad < 0 则为逆时针旋转
    Args:
        src_point:
        rot_rad:

    Returns:

    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_dir_2(src_point, rot_rad):
    sn, cs = np.abs(np.sin(rot_rad)), np.abs(np.cos(rot_rad))

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs + src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # left 表示中心点到 masked 区域最左边的距离
    # right 表示中心点到 masked 区域最右边的距离
    left, right = min(x, radius), min(width - x, radius + 1)
    # top 表示中心点到 masked 区域最上边的距离
    # bottom 表示中心点到 masked 区域最下边的距离
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # 这里就是同一个类的多个目标的高斯核发生重叠取最大值
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def test_affine(image, rot_angle=0):
    h, w = image.shape[:2]
    src_image = image.copy()
    dst_h, dst_w = 512, 512
    center = np.array((w // 2, h // 2))
    scale = max(h, w)
    trans_matrix, src, dst = get_affine_transform(center, scale, rot_angle, (512, 512))
    dst_image = cv2.warpAffine(src_image, trans_matrix, (dst_h, dst_w), flags=cv2.INTER_LINEAR)

    src_image = np.pad(src_image, ((200, 0), (200, 0), (0, 0)), mode='constant')
    src[:, 0] += 200
    src[:, 1] += 200
    for p in src:
        cv2.circle(src_image, tuple(p.astype(np.int32)), 10, (0, 255, 0), -1)
    for p in dst:
        cv2.circle(dst_image, tuple(p.astype(np.int32)), 10, (0, 0, 255), -1)
    cv2.namedWindow('src_image_{}'.format(rot_angle), cv2.WINDOW_NORMAL)
    cv2.imshow('src_image_{}'.format(rot_angle), src_image)
    cv2.namedWindow('dst_image_{}'.format(rot_angle), cv2.WINDOW_NORMAL)
    cv2.imshow('dst_image_{}'.format(rot_angle), dst_image)


if __name__ == '__main__':
    ##########################################
    # learn get_dir
    ##########################################
    # p1 = (200, 250)
    # p2 = (400, 250)
    # p3 = (400, 350)
    # p4 = (200, 350)
    # p1_ = get_dir(p1, 0.5236)
    # p2_ = get_dir(p2, 0.5236)
    # p3_ = get_dir(p3, 0.5236)
    # p4_ = get_dir(p4, 0.5236)
    # p1__ = get_dir_2(p1, 0.5236)
    # p2__ = get_dir_2(p2, 0.5236)
    # p3__ = get_dir_2(p3, 0.5236)
    # p4__ = get_dir_2(p4, 0.5236)
    # image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    # for p in [p1, p2, p3, p4]:
    #     cv2.circle(image, p, 10, (0, 255, 0), -1)
    # for p in [p1_, p2_, p3_, p4_]:
    #     cv2.circle(image, tuple(map(int, p)), 10, (255, 255, 0), -1)
    # for p in [p1__, p2__, p3__, p4__]:
    #     cv2.circle(image, tuple(map(int, p)), 10, (0, 255, 255), -1)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    #############################################
    # learn get_affine_transform
    #############################################
    image = cv2.imread('data/voc/images/000012.jpg')
    test_affine(image, 0)
    test_affine(image, 30)
    cv2.waitKey(0)
