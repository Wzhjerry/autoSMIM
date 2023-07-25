import glob
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import slic

cv2.setNumThreads(1)


def save_np_full(corenal_list):

    sp_num = 512  # change if needed
    size = 512  # change if needed

    raw = cv2.imread(corenal_list)
    mask_num = corenal_list.split("/")[-1][:-4]
    raw = cv2.resize(raw, (size, size))

    Path(f"/data/share/wangzh/datasets/Ham10000/T_sp_{sp_num}/").mkdir(parents=True, exist_ok=True)
    Path(f'/data/share/wangzh/datasets/Ham10000/T_sp_{sp_num}_preview/').mkdir(parents=True, exist_ok=True)

    segments = slic(raw, n_segments=sp_num, compactness=10, start_label=1)
    np.save(
        f"/data/share/wangzh/datasets/Ham10000/T_sp_{sp_num}/{mask_num}",
        np.uint16(segments),
    )
    # save = mark_boundaries(raw, segments, color=(0, 0, 1)) * 255
    # cv2.imwrite(f'/data/share/wangzh/datasets/Ham10000/T_sp_{sp_num}_preview/{mask_num}.png', save)
    print(corenal_list + " done")


if __name__ == "__main__":
    list_all = []
    # need modification
    corenal_list = glob.glob("/data/share/wangzh/datasets/Ham10000/part1/*.jpg")
    corenal_list.sort()

    # need modification
    corenal_list2 = glob.glob("/data/share/wangzh/datasets/Ham10000/part2/*.jpg")
    corenal_list2.sort()

    with Pool(processes=60) as pool:
        data_processed = pool.map(save_np_full, corenal_list)
    with Pool(processes=60) as pool:
        data_processed = pool.map(save_np_full, corenal_list2)
