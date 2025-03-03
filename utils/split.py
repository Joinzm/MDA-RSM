import os
import glob
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time


def patch_format(inp):
    img_path, mask_path, imgs_output_dir, masks_output_dir = inp

    # 读取影像和掩膜
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    id = os.path.splitext(os.path.basename(img_path))[0]

    # 确保影像和掩膜尺寸一致
    assert img.shape == mask.shape, f"Shape mismatch for {img_path} and {mask_path}"
    assert img.shape[:2] == (1500, 1500), f"{img_path} is not 1500x1500, skipping."

    # 定义裁剪坐标
    crop_coords = [
        (0, 1024, 0, 1024),  # 左上角
        (0, 1024, 476, 1500),  # 右上角
        (476, 1500, 0, 1024),  # 左下角
        (476, 1500, 476, 1500),  # 右下角
    ]

    for idx, (y_start, y_end, x_start, x_end) in enumerate(crop_coords):
        img_tile = img[y_start:y_end, x_start:x_end]
        mask_tile = mask[y_start:y_end, x_start:x_end]

        # 保存裁剪后的影像和掩膜
        out_img_path = os.path.join(imgs_output_dir, f"{id}_{idx}.png")
        out_mask_path = os.path.join(masks_output_dir, f"{id}_{idx}.png")
        cv2.imwrite(out_img_path, img_tile)
        cv2.imwrite(out_mask_path, mask_tile)


if __name__ == "__main__":
    # 输入输出路径
    input_img_dir = "/home/data/MASS_Building/val"
    input_mask_dir = "/home/data/MASS_Building/val_labels"
    output_img_dir = "/home/data/MASS_Building_1024/val/image"
    output_mask_dir = "/home/data/MASS_Building_1024/val/label"

    # 获取文件列表
    img_paths = sorted(glob.glob(os.path.join(input_img_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(input_mask_dir, "*.png")))

    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 准备并行处理的输入
    inp = [
        (img_path, mask_path, output_img_dir, output_mask_dir)
        for img_path, mask_path in zip(img_paths, mask_paths)
    ]

    # 并行处理裁剪
    t0 = time.time()
    mpp.Pool(processes=5).map(patch_format, inp)
    t1 = time.time()

    print(f"影像裁剪完成，用时: {t1 - t0:.2f} 秒")
