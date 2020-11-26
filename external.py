import os
import numpy as np
import scipy.linalg as linalg


# 设计成读取单个txt的函数，更加灵活
# 因为在求解外参的时候，需要不同的H
# 每个H和内参A一一对应出一个外参

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
    print('求解外参初始值')

    # 读取内参A
    A = read_txt('A.txt')
    print('内参A：\n', A)

    # 外参初始化
    Rt = np.zeros((3, 4))

    # 读取H单应矩阵
    H_dir_list = os.listdir('Hbuf')  # 所有的H文件名
    num_H = len(H_dir_list)
    for i in range(num_H):
        H_mat = read_txt('Hbuf/' + H_dir_list[i])
        print('H_mat:\n', H_mat)  # ok

        lamada = 1 / np.linalg.norm(np.dot(linalg.inv(A), H_mat[:, 0:1]))
        r1 = lamada * np.dot(linalg.inv(A), H_mat[:, 0:1]).reshape(3)  # H_mat1
        r2 = lamada * np.dot(linalg.inv(A), H_mat[:, 1:2]).reshape(3)  # H_mat2\
        r3 = np.cross(r1, r2)  # 要保证满足右手坐标系
        t = lamada * np.dot(linalg.inv(A), H_mat[:, 2:3])  # h3

        print('Orthogonal?', np.dot(r1, r2))  # 发现点乘！=0 需要正交化处理
        # R = [r1, r2, r3]
        # u, s, v = np.linalg.svd(R)
        # R =

        # print('r1:\n', r1)
        # print('r2:\n', r2)
        # print('r3:\n', r3)
        # print('t:\n', t)
        Rt[:, 0:1] = r1.reshape(3, 1)
        Rt[:, 1:2] = r2.reshape(3, 1)
        Rt[:, 2:3] = r3.reshape(3, 1)

        u, s, v = np.linalg.svd(Rt[:, 0:3])  # 对旋转矩阵正交化
        Rt[:, 0:3] = np.dot(u, v)  # v是已经转置的了

        Rt[:, 3:4] = t
        np.set_printoptions(suppress=True)  # 取消使用科学技术法
        print('Rt:\n', Rt)  # 3x4

        np.savetxt('RtBuf/Rt_' + H_dir_list[i] + '.txt', Rt)


    # 求畸变系数


    # 优化 提高精度
