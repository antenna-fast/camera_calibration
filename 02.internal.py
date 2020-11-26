import os
import numpy as np
import scipy.linalg as linalg


def read_txt(file_path):
    # 每次读取一个H文件，返回H矩阵

    H = np.zeros((3, 3))

    f = open(file_path)
    lines = f.readlines()
    # print(lines)

    for i in range(len(lines)):  # 行
        line = lines[i].split(' ')
        # print('line:', line)
        for j in range(len(line)):  # 列
            H[i][j] = float(line[j].strip())
    return H


if __name__ == '__main__':
    H_dir_list = os.listdir('Hbuf')  # 所有的H文件名
    num_H = len(H_dir_list)

    V = []

    for i in range(num_H):
        H_mat = read_txt('Hbuf/' + H_dir_list[i])
        # print(H_mat)  # ok

        v11 = [H_mat[0][0] * H_mat[0][0],  # 1  00*00
               H_mat[0][0] * H_mat[1][0] + H_mat[1][0] * H_mat[0][0],  # 2 00*10 + 10*00
               H_mat[1][0] * H_mat[1][0],  # 3  10*10
               H_mat[2][0] * H_mat[0][0] + H_mat[0][0] * H_mat[2][0],  # 4  20*00 + 00*20
               H_mat[2][0] * H_mat[1][0] + H_mat[1][0] * H_mat[2][0],  # 5  20*10 + 10*20
               H_mat[2][0] * H_mat[2][0]  # 6  20*20
               ]

        v12 = [H_mat[0][0] * H_mat[0][1],  # 1  00*01
               H_mat[0][0] * H_mat[1][1] + H_mat[1][0] * H_mat[0][1],  # 2 00*11 + 10*01
               H_mat[1][0] * H_mat[1][1],  # 3  10*11
               H_mat[2][0] * H_mat[0][1] + H_mat[0][0] * H_mat[2][1],  # 4  20*01 + 00*21
               H_mat[2][0] * H_mat[1][1] + H_mat[1][0] * H_mat[2][1],  # 5  20*11 + 10*21
               H_mat[2][0] * H_mat[2][1]  # 6  20*21
               ]

        v22 = [H_mat[0][1] * H_mat[0][1],  # 1  00*00  前i后j
               H_mat[0][1] * H_mat[1][1] + H_mat[1][1] * H_mat[0][1],  # 2 00*10 + 10*00
               H_mat[1][1] * H_mat[1][1],  # 3  10*10
               H_mat[2][1] * H_mat[0][1] + H_mat[0][1] * H_mat[2][1],  # 4  20*00 + 00*20
               H_mat[2][1] * H_mat[1][1] + H_mat[1][1] * H_mat[2][1],  # 5  20*10 + 10*20
               H_mat[2][1] * H_mat[2][1]  # 6  20*20
               ]

        # 用vij构造V
        V.append(v12)
        V.append(np.array(v11) - np.array(v22))

    V = np.array(V)
    print('V:\n', V)
    # print('V.shape:', V.shape)  # (12, 6)

    # 对V进行SVD 求出b  求出来的b不一样

    u, s, v = np.linalg.svd(V)
    # u, s, v = np.linalg.svd(np.dot(V.T, V))
    # print('s:{0} \n  {1}\n  {2} \n'.format(u, s, v))

    print('v:\n', v.shape)  # b[b11-0  b12-1  b22-2  b13-3  b23-4  b33-5]
    print('v:\n', v)
    print('s:\n', s)

    b = v.T[:, 5]  # 最后一列是Ax=0的最小二乘的解  注意是v.T !! 因为苦寒暑求出来转置了一下
    print('b:', b)
    # 从b构建内参  附录B P18
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)  # -0.001658921!!! 显然不对
    lama = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = pow(lama / b[0], 0.5)
    beta = pow(lama * b[0] / (b[0] * b[2] - b[1] ** 2), 0.5)
    gama = -1 * b[1] * (alpha ** 2) * beta / lama
    u0 = gama * v0 / beta - b[3] * (alpha ** 2) / lama  # Nan!!!

    # print('前面一堆:', (b[1] * b[3] - b[0] * b[4]))  # 1.5018854251130593e-05
    # print('后面一堆:', (b[0] * b[2] - b[1] ** 2))  # 后面一堆: -0.009053382678418754
    # print('beta:', beta)  # 0.05149250076804658
    # print('lama:', lama)  # lama: -2.4006229171703326e-05  太小了!!
    # 问题解决！没有仔细阅读svd库函数的输出，因为要转置才对
    print('u0: {0},  v0:{1}'.format(u0, v0))

    A = np.array([[alpha, gama, u0],
                  [0, beta, v0],
                  [0, 0, 1]])

    np.set_printoptions(suppress=True)  # 取消使用科学技术法
    print('内参矩阵:\n', A)

    # 保存
    np.savetxt('A.txt', A)
