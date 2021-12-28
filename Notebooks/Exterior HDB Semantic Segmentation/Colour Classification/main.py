import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def convert(path):
    def check_grey(bgr_values):
        return (np.abs(int(bgr_values[0]) - int(bgr_values[1])) +
                np.abs(int(bgr_values[1]) - int(bgr_values[2])) +
                np.abs(int(bgr_values[0]) - int(bgr_values[2]))) < 20

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img1 = img.copy()
    img1[np.all(img1 == [0, 0, 0, 0], axis=-1)] = [255, 255, 255, 255]
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)

    img1[np.any(img1 != [255, 255, 255, 255], axis=-1)] = [0, 0, 0, 0]
    img3 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if(check_grey(img2[i, j])):
                img2[i, j] = np.array([127, 127, 127])

    img4 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img4[(img4[:, :, 0] < 70) & (img4[:, :, 0] > 40)] = np.array([60, 255, 255])
    img5 = cv2.cvtColor(img4, cv2.COLOR_HSV2BGR)
    img5[np.any((img5 != [0, 255, 0]) & (
        img5 != [127, 127, 127]), axis=-1)] = [0, 0, 255]
    img5[np.all(img3 == [255, 255, 255], axis=-1)] = [255, 255, 255]
    cv2.imwrite(path.rename("annotated", "annotations"), img5)


for i,path in enumerate(glob("snapshots/annotated/*.png")):
    print(f"{i}:",end=" ")
    convert(path)
    print(path)
