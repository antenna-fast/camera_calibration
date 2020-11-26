# 畸变矫正

import numpy as np

import os

def read_txt(file_path, r=3, c=3):  # 文件名 行 列
    #if file_path == None:
    #    raise False

    # 每次读取一个H文件，返回H矩阵
    H = np.zeros((r, c))

    f = open(file_path)
    lines = f.readlines()
    # print(lines)

    for i in range(len(lines)):  # 行
        line = lines[i].split(' ')
        # print('line:', line)
        for j in range(len(line)):  # 列
            H[i][j] = float(line[j].strip())
    return H


if __name__ == "__main__":

    # 读取内参文件
    A = read_txt("D:/SIA/VS/Python/CamCalibration/A.txt", 3, 3)
    print("A:\n", A)

    # 读取外参文件
    rts_path = "D:/SIA/VS/Python/CamCalibration/RtBuf/"
    Rts = os.listdir(rts_path)
    #print(Rts)

    u0 = A[2][0]
    v0 = A[2][1]

    for i in range(len(Rts)):  # num of img
        f_rt = Rts[i]
        #print(f_rt)

        rt = read_txt(rts_path+f_rt, 3, 4)
        # print("rt:\n", rt)
        # Internal * External
        i_e = np.dot(A, rt)

        # 读取该外参文件对应的所有点
        pts_file = f_rt[5:]
        print(pts_file)
        a = "D:/SIA/VS/Python/CamCalibration/数据/"
        #print(dir)
        #print("pts_shape:", pts.shape)
        #a = np.loadtxt("数据/内参/"+pts_file)
        #print(a)