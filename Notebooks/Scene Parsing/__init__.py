import pathlib
import numpy as np
import pandas as pd
import cv2

file_dir = pathlib.Path(__file__).parent.resolve()


def convertFromADE(img):
    indexMapping = pd.DataFrame(np.loadtxt(file_dir / "mapFromADE.txt", int)).rename(columns={0: "clr", 1: "cls"}).set_index("cls").clr
    h, w, _ = img.shape
    h, w = (512, round(w/h*512)) if 512 < h < w else (round(h/w*512),
                                                      512) if 512 < w < h else (h, w)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)[:, :, ::-1]
    ade = np.uint64(np.uint16(img[:, :, 0])/10*256 + np.uint16(img[:, :, 1]))
    out = np.zeros_like(ade, dtype=np.uint8)
    for cls in set(np.unique(ade.flatten())) & set(indexMapping.index):
        out[ade == cls] = indexMapping.loc[cls]
    return out
