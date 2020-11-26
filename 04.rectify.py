# �������

import numpy as np

import os

def read_txt(file_path, r=3, c=3):  # �ļ��� �� ��
    #if file_path == None:
    #    raise False

    # ÿ�ζ�ȡһ��H�ļ�������H����
    H = np.zeros((r, c))

    f = open(file_path)
    lines = f.readlines()
    # print(lines)

    for i in range(len(lines)):  # ��
        line = lines[i].split(' ')
        # print('line:', line)
        for j in range(len(line)):  # ��
            H[i][j] = float(line[j].strip())
    return H


if __name__ == "__main__":

    # ��ȡ�ڲ��ļ�
    A = read_txt("D:/SIA/VS/Python/CamCalibration/A.txt", 3, 3)
    print("A:\n", A)

    # ��ȡ����ļ�
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

        # ��ȡ������ļ���Ӧ�����е�
        pts_file = f_rt[5:]
        print(pts_file)
        a = "D:/SIA/VS/Python/CamCalibration/����/"
        #print(dir)
        #print("pts_shape:", pts.shape)
        #a = np.loadtxt("����/�ڲ�/"+pts_file)
        #print(a)