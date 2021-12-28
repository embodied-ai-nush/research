import sys, cv2
import pathlib
from . import convertFromADE

file_dir = pathlib.Path(__file__).parent.resolve()

#convertFromADE("ADE_train_00000970_raw.png", "ADE_train_00000970_challenge.png")

if __name__ == "__main__":
    i, f = tuple(sys.argv[:2]) if len(sys.argv) else (file_dir / "ADE_train_00000970_raw.png", file_dir / "ADE_train_00000970_challenge.png")
    img = cv2.imread(i)
    out = convertFromADE(i)
    cv2.imwrite(f, out)
