import cv2
import numpy as np

# 換成您的其中一張 GT 圖片路徑
gt_path = "/home/rvl1421/Datasets/Cityscapes/GT/gtFine/train/bochum/bochum_000000_000313_gtFine_labelTrainIds.png"

img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
unique_values = np.unique(img)

print(f"圖片中的像素值: {unique_values}")

# 判斷標準：
# 如果您看到 7, 26, 33 這種數字 -> ❌ 這是原始 ID，需要轉換。
# 如果您只看到 0~18 (以及可能的 255) -> ✅ 這是正確的 Train ID。