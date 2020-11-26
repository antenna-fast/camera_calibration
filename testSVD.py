import numpy as np
import re
from scipy.linalg import svd
import cv2


# 读取从V过来的矩阵
def read_txt(file_path):
    # 返回H矩阵

    H = np.zeros((12, 6))

    f = open(file_path)

    lines = f.readlines()
    # print(lines)

    for i in range(len(lines)):  # 行
        line = lines[i].split(' ')

        # 过滤掉空字符 非常好用
        line = list(filter(None, line))
        # print('line:', line)
        for j in range(len(line)):  # 列
            H[i][j] = float(line[j].strip())
    return H


if __name__ == '__main__':
    V = read_txt('V.txt')
    print('V:\n', V)

    # np
    u, s, v = np.linalg.svd(V.astype(np.double))
    # u, s, v = np.linalg.svd(np.dot(V.T, V))

    # scipy
    # u2, s2, v2 = svd(V)

    print('v:\n', v)
    # print('v2:\n', v2)

    # opencv
    # res1 = cv2.SVDecomp(V)
    # res1 = cv2.SVDecomp(np.dot(V.T, V))
    # print('res:\n', res1)