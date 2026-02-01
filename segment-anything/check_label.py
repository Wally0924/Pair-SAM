import cv2
import numpy as np

# 換成你的 GT 路徑
gt_path = "/home/rvl1421/Datasets/Cityscapes/GT/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"
img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

print(f"圖片中的所有數值: {np.unique(img)}")