import numpy as np
import sys
from torch.utils.data import DataLoader
import torch
import logging
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
)

# from utils.data_loading import BasicDataset
from utils.mass_dataset import *
from config.hyperparameter_mdscans_MASS import ph
from model.mda_rsm import MDA_RSM
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import random
import ttach as tta


def random_seed(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = (
        True  # keep convolution algorithm deterministic
    )
    torch.backends.cudnn.benchmark = (
        False  # using fixed convolution algorithm to accelerate training
    )
    # if model and input are fixed, set True to search better convolution algorithm
    # torch.backends.cudnn.benchmark = True


def auto_inference():
    random_seed(SEED=ph.random_seed)
    try:
        test_net(ph)
    except KeyboardInterrupt:
        logging.info("Error")
        sys.exit(0)


def test_net(ph, load_checkpoint=True):
    # output路径
    output_path = f"/home/view/{ph.project_name}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # 1. Create dataset

    # test_dataset = BasicDataset(images_dir=f'./{dataset_name}/test/image/',
    #                             labels_dir=f'./{dataset_name}/test/label/',
    # train=False)
    test_dataset = MassBuildDataset(
        data_root=f"{ph.root_dir}",
        mode="test",
        img_dir=f"{ph.dataset_name}/test/image",
        mask_dir=f"{ph.dataset_name}/test/label",
        transform=get_validation_transform(),
    )
    # 2. Create data loaders
    loader_args = dict(num_workers=2, prefetch_factor=5, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=ph.batch_size,
        **loader_args,
    )
    #  batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device {device}")

    net = MDA_RSM(
        dims=ph.dims,
        depths=ph.depths,
        ssm_d_state=ph.ssm_d_state,
        ssm_dt_rank=ph.ssm_dt_rank,
        ssm_ratio=ph.ssm_ratio,
        mlp_ratio=ph.mlp_ratio,
        image_size=ph.image_size,
        scan_types=ph.scan_types,
        per_scan_num=ph.per_scan_num,
    )
    net.to(device=device)

    assert ph.load, "Loading model error, checkpoint ph.load"
    load_model = torch.load(ph.load, map_location=device)
    if "best" not in ph.load:
        load_model = load_model["net"]
    if load_checkpoint:
        # net.load_state_dict(load_model['net'])
        # print(load_model)
        net.load_state_dict(load_model, strict=False)
    else:
        net.load_state_dict(load_model, strict=False)
    logging.info(f"Model loaded from {ph.load}")
    # torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection(
        {
            "accuracy": Accuracy().to(device=device),
            "precision": Precision().to(device=device),
            "recall": Recall().to(device=device),
            "f1score": F1Score().to(device=device),
        }
    )  # metrics calculator

    net.eval()

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 90, 180, 270])
        ]
    )
    net = tta.SegmentationTTAWrapper(net, transforms)

    logging.info("SET model mode to test!")

    # def compute_iou(preds, labels):
    #     """
    #     计算 IoU（Intersection over Union）
    #     :param preds: 预测值，形状 [B, H, W] 或 [B, 1, H, W]，预测的概率或标签。
    #     :param labels: 真实标签，形状 [B, H, W] 或 [B, 1, H, W]。
    #     :param threshold: IoU 计算的阈值，通常使用 0.5 作为二分类的阈值。
    #     :return: 每个批次的 IoU。
    #     """
    #     # 计算TP、FP、FN、TN
    #     TP = torch.sum((preds == 1) & (labels == 1)).float()
    #     FP = torch.sum((preds == 1) & (labels == 0)).float()
    #     FN = torch.sum((preds == 0) & (labels == 1)).float()
    #     TN = torch.sum((preds == 0) & (labels == 0)).float()

    #     # 计算IoU
    #     iou = TP / (TP + FP + FN + 1e-6)  # 加上1e-6避免除0错误

    #     return iou

    with torch.no_grad():
        for batch_img1, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            labels = labels.float().to(device)

            ss_preds = net(batch_img1)
            ss_preds = torch.sigmoid(ss_preds)

            # Calculate and log other batch metrics
            ss_preds = ss_preds.float()

            for i in range(0, len(name)):
                pre_cpu = ss_preds.squeeze().cpu().numpy()
                pre = (pre_cpu[i] > 0.5).astype(np.uint8) * 255
                mask_pth = f"{output_path}/{name[i]}.png"
                cv2.imwrite(mask_pth, pre)

            labels = labels.int().unsqueeze(1)
            metric_collection.update((ss_preds > 0.5).float(), labels)
            # metric_collection.update(ss_preds, labels)

            # clear batch variables from memory
            del batch_img1, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print("over")


if __name__ == "__main__":

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    auto_inference()
