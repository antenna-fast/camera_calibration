import numpy as np
from scipy.spatial.transform import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('RtTest')

    a = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0]])

    eu = [0, 90, 0]
    # f = Rotation.from_euler('zxy', eu, degrees=True)
    f = Rotation.from_euler('xyz', eu, degrees=True)
    g = f.as_matrix()

    # Rotation(a)
    a_trans = np.dot(g, a)
    print(a_trans)