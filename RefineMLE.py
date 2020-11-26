import os
import numpy as np


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


# 读取世界点以及投影后的点，读取内参外参进行投影
# 使用MLE进行refine优化
if __name__ == '__main__':
    print('RefineMLE')

    # 读取内参A
    A = read_txt('A.txt')
    print('内参A：\n', A)

    # 读取每个H
    # 初始化
    Rt = np.zeros((3, 4))

    # 读取所有的点、内参以及对应的Rt，进行投影
    files = os.listdir('数据/内参')

    H_dir_list = os.listdir('Hbuf')  # 所有的H文件名
    num_H = len(H_dir_list)

    # for i in range(num_H):
    #     H_mat = read_txt('Hbuf/' + H_dir_list[i])
    #     print('H_mat:\n', H_mat)  # ok

    for file_name in files:  # 不同文件对应不同的但应矩阵，所以需要分别求解

        points_img_x = []
        points_img_y = []

        points_w_x = []
        points_w_y = []
        points_w_z = []


        print('file_name:', file_name)
        ofile = open('数据/内参/' + file_name)
        lines = ofile.readlines()

        for line in lines:  # 每一行
            # points_list.append(float(line[:3]))  # 序号  将就opencv的函数
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
