import os

import numpy as np
import cv2
from scipy import linalg


if __name__ == '__main__':

    print('单应矩阵')

    # 读取内参数据
    # 所有文件
    files = os.listdir('数据/内参')
    # print(files)

    V = []
    H_buf = []

    for file_name in files:  # 不同文件对应不同的但应矩阵，所以需要分别求解

        points_img_x = []
        points_img_y = []

        points_w_x = []
        points_w_y = []
        points_w_z = []

        print(file_name)
        ofile = open('数据/内参/' + file_name)
        lines = ofile.readlines()

        A = []
        b = []

        for line in lines:  # 每一行
            # points_list.append(float(line[:3]))  # 序号
            points_img_x.append(float(line[6:17]))
            points_img_y.append(float(line[19:32]))
            points_w_x.append(float(line[33:44]))
            points_w_y.append(float(line[45:57]))
            points_w_z.append(float(line[59:69]))

            # 第二种格式 直接将点弄到A和b里面
            point_img_x = float(line[6:17])
            point_img_y = float(line[19:32])
            point_w_x = float(line[33:44])
            point_w_y = float(line[45:57])
            #
            A.append([point_w_x, point_w_y, 1, 0, 0, 0, -1 * point_w_x * point_img_x, -1 * point_w_y * point_img_x])
            A.append([0, 0, 0, point_w_x, point_w_y, 1, -1 * point_w_x * point_img_y, -1 * point_w_y * point_img_y])
            #
            b.append(point_img_x)
            b.append(point_img_y)

        points_img = np.array([points_img_x, points_img_y]).transpose()
        # points_w = np.array([points_w_x, points_w_y, points_w_z]).transpose()
        points_w = np.array([points_w_x, points_w_y]).transpose()

        # 和opencv的函数作对比 说明H求的没问题
        # srcPoints, dstPoints, method=None, ransacReprojThreshold=None, mask=None, maxIters=None, confidence=None
        h = cv2.findHomography(points_w, points_img, method=cv2.RANSAC)
        print('h:', h[0])

        # print('世界坐标：\n', points_w)
        # print('图像坐标：\n', points_img)

        print(len(points_img))
        print(len(points_w))

        A = np.array(A)
        b = np.array(b)

        print('A.shape:', A.shape)
        print('b.shape:', b.shape)

        x = np.linalg.lstsq(A, b, rcond=None)
        x = x[0]
        print('x:\n', x)

        # print('re-project test Ax=:\n', np.dot(A, x[0]))
        # print('b=', b)

        # 恢复H的形状
        H = np.array([[x[0], x[1], x[2]],
                      [x[3], x[4], x[5]],
                      [x[6], x[7], 1]])

        print('H:\n', H)

        H_buf.append(H)

        # 利用外参的旋转矩阵约束 从H恢复内参
        # 直接构建Vij
        v_ij = np.zeros((3, 3, 6))  # vij有6列

        for i in range(3):
            for j in range(3):
                v_ij[i, j] = np.array([H[i, 0] * H[j, 0],
                                       H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0],
                                       H[i, 1] * H[j, 1],
                                       H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2],
                                       H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2],
                                       H[i, 2] * H[j, 2]
                                       ])
        # # print('v_ij', v_ij)

        # '''
        #         v11 = [H[0, 0] * H[0, 0],
        #                H[0, 0] * H[0, 1] + H[0, 1] * H[0, 0],
        #                H[0, 1] * H[0, 1],
        #                H[0, 2] * H[0, 0] + H[0, 0] * H[0, 2],
        #                H[0, 2] * H[0, 1] + H[0, 1] * H[0, 2],
        #                H[0, 2] * H[0, 2]
        #                ]
        #
        #         v12 = [H[0, 0] * H[0, 0],
        #                H[0, 0] * H[0, 1] + H[0, 1] * H[0, 0],
        #                H[0, 1] * H[0, 1],
        #                H[0, 2] * H[0, 0] + H[0, 0] * H[0, 2],
        #                H[0, 2] * H[0, 1] + H[0, 1] * H[0, 2],
        #                H[0, 2] * H[0, 2]
        #                ]
        #
        #         v22 = [H[0, 0] * H[0, 0],
        #                H[0, 0] * H[0, 1] + H[0, 1] * H[0, 0],
        #                H[0, 1] * H[0, 1],
        #                H[0, 2] * H[0, 0] + H[0, 0] * H[0, 2],
        #                H[0, 2] * H[0, 1] + H[0, 1] * H[0, 2],
        #                H[0, 2] * H[0, 2]
        #                ]
        # '''

        # 求b

        V.append(v_ij[0, 1])
        V.append((v_ij[0, 0] - v_ij[1, 1]))

    V = np.array(V)
    # 求Vb=0中的b 这是内参的函数
    # SVD
    print('V:', V.shape)  # 2*n个方程
    # print('V:', V)  # 2*n个方程

    # u, s, v = np.linalg.svd(V)
    u, s, v = linalg.svd(V)
    # print('s:{0} \n  {1}\n  {2} \n'.format(u, s, v))
    print('v:\n', v.shape)  # b[b11-0  b12-1  b22-2  b13-3  b23-4  b33-5]

    b = v[:, 5]  # 最后一列是Ax=0的最小二乘的解
    # b = v[5]
    # 从b构建内参  附录B P18
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)  # -0.0016589218399981113!!! 显然不对
    lama = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = pow(lama / b[0], 0.5)
    beta = pow(lama * b[0] / (b[0] * b[2] - b[1] ** 2), 0.5)
    gama = -1 * b[1] * (alpha ** 2) * beta / lama
    u0 = gama * v0 / beta - b[3] * (alpha ** 2) / lama  # Nan!!!

    # print('前面一堆:', (b[1] * b[3] - b[0] * b[4]))  # 1.5018854251130593e-05
    # print('后面一堆:', (b[0] * b[2] - b[1] ** 2))  # 后面一堆: -0.009053382678418754
    # print('beta:', beta)  # 0.05149250076804658
    # print('lama:', lama)  # lama: -2.4006229171703326e-05  太小了!!
    print('u0: {0},  v0:{1}'.format(u0, v0))

    A = np.array([[alpha, gama, u0],
                  [0, beta, v0],
                  [0, 0, 1]])


