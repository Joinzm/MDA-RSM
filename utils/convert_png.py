from PIL import Image
import os
import multiprocessing


def convert_tif_to_png(tif_path, png_path):
    """Convert a single TIFF file to PNG format."""
    try:
        with Image.open(tif_path) as img:
            img.save(png_path, format="PNG")
            print(
                f"Converted {os.path.basename(tif_path)} to {os.path.basename(png_path)}"
            )
    except Exception as e:
        print(f"Failed to convert {tif_path}: {e}")


def convert_all_tifs(tif_folder, png_folder):
    """Convert all TIFF files in the specified folder to PNG format using multiprocessing."""
    # 创建输出文件夹（如果不存在）
    os.makedirs(png_folder, exist_ok=True)

    # 收集所有 TIFF 文件的路径
    tif_files = [
        os.path.join(tif_folder, filename)
        for filename in os.listdir(tif_folder)
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff")
    ]

    # 准备 PNG 文件的路径
    png_files = [
        os.path.join(
            png_folder, f"{os.path.splitext(os.path.basename(tif_file))[0]}.png"
        )
        for tif_file in tif_files
    ]

    # 创建进程池
    with multiprocessing.Pool(processes=64) as pool:
        pool.starmap(convert_tif_to_png, zip(tif_files, png_files))
