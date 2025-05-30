import sys
import time
import numpy as np
from torch import optim
import torchvision.transforms as T

# from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
)
import os
import logging
import random
import wandb
from utils.losses import FCCDN_loss_without_seg
from utils.utils import train_val_test
from utils.mass_dataset import *

# from utils.data_loading import BasicDataset
from config.hyperparameter_mdscans_MASS import ph
from model.mda_rsm import MDA_RSM


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


def train_net(dataset_name):
    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val` /`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """
    # 1. Create dataset, checkpoint and best model path

    # dataset path should be dataset_name/train or val/t1 or t2 or label

    # define the dataloader
    train_dataset = MassBuildDataset(
        data_root=f"{ph.root_dir}",
        img_dir=f"{dataset_name}/train/image",
        mask_dir=f"{dataset_name}/train/label",
        mosaic_ratio=0.25,
        transform=get_training_transform(),
    )

    val_dataset = MassBuildDataset(
        data_root=f"{ph.root_dir}",
        mode="val",
        img_dir=f"{dataset_name}/val/image",
        mask_dir=f"{dataset_name}/val/label",
        transform=get_validation_transform(),
    )
    # train_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/train/image/',
    #                              labels_dir=f'{ph.root_dir}/{dataset_name}/train/label/',
    #                              train=True)
    # val_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/val/image/',
    #                            labels_dir=f'{ph.root_dir}/{dataset_name}/val/label/',
    #                            train=False)
    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(
        num_workers=2,
        prefetch_factor=5,
        persistent_workers=True,
    )
    # loader_args = dict(num_workers=2,  # 尝试设置为0来测试是否是多进程问题
    #                prefetch_factor=None,  # 降低预取因子
    #                persistent_workers=False,  # 关闭持久化工作进程
    #                )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        batch_size=ph.batch_size,
        **loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=ph.batch_size,
        **loader_args,
    )

    # 4. Initialize logging

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict["time"] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous
    log_wandb = wandb.init(
        project=ph.log_wandb_project,
        resume="allow",
        anonymous="must",
        settings=wandb.Settings(start_method="thread"),
        config=hyperparameter_dict,
        mode="offline",
    )
    import os

    os.environ["WANDB_DIR"] = f"./{ph.log_wandb_project}"

    logging.info(
        f"""Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
    """
    )

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things
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
    )  # change detection model
    net = net.to(device=device)
    optimizer = optim.AdamW(
        net.parameters(), lr=ph.learning_rate, weight_decay=ph.weight_decay
    )  # optimizer
    warmup_lr = np.arange(
        1e-7, ph.learning_rate, (ph.learning_rate - 1e-7) / ph.warm_up_step
    )  # warm up learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
    #                                                  factor=ph.factor)  # learning rate scheduler
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint["net"])
        logging.info(f"Model loaded from {ph.load}")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            for g in optimizer.param_groups:
                g["lr"] = ph.learning_rate
            optimizer.param_groups[0]["capturable"] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    criterion = FCCDN_loss_without_seg  # loss function

    best_metrics = dict.fromkeys(
        ["best_f1score", "lowest loss"], 0
    )  # best evaluation metrics
    metric_collection = MetricCollection(
        {
            "accuracy": Accuracy().to(device=device),
            "precision": Precision().to(device=device),
            "recall": Recall().to(device=device),
            "f1score": F1Score().to(device=device),
        }
    )  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    checkpoint_path = f"checkpoint/{ph.project_name}/checkpoint/"
    best_f1score_model_path = f"checkpoint/{ph.project_name}/best_f1score_model/"
    best_loss_model_path = f"checkpoint/{ph.project_name}/best_loss_model/"

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(best_f1score_model_path, exist_ok=True)
    os.makedirs(best_loss_model_path, exist_ok=True)

    non_improved_epoch = (
        0  # adjust learning rate when non_improved_epoch equal to patience
    )

    # 5. Begin training

    for epoch in range(ph.epochs):

        print("Start Train!")

        log_wandb, net, optimizer, grad_scaler, total_step, lr = train_val_test(
            ph=ph,
            mode="train",
            dataset_name=dataset_name,
            dataloader=train_loader,
            device=device,
            log_wandb=log_wandb,
            net=net,
            optimizer=optimizer,
            total_step=total_step,
            lr=lr,
            criterion=criterion,
            metric_collection=metric_collection,
            to_pilimg=to_pilimg,
            epoch=epoch,
            warmup_lr=warmup_lr,
            grad_scaler=grad_scaler,
        )

        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        if (epoch + 1) >= ph.evaluate_epoch and (epoch + 1) % ph.evaluate_inteval == 0:
            print("Start Validation!")

            with torch.no_grad():
                (
                    log_wandb,
                    net,
                    optimizer,
                    total_step,
                    lr,
                    best_metrics,
                    non_improved_epoch,
                ) = train_val_test(
                    ph=ph,
                    mode="val",
                    dataset_name=dataset_name,
                    dataloader=val_loader,
                    device=device,
                    log_wandb=log_wandb,
                    net=net,
                    optimizer=optimizer,
                    total_step=total_step,
                    lr=lr,
                    criterion=criterion,
                    metric_collection=metric_collection,
                    to_pilimg=to_pilimg,
                    epoch=epoch,
                    best_metrics=best_metrics,
                    checkpoint_path=checkpoint_path,
                    best_f1score_model_path=best_f1score_model_path,
                    best_loss_model_path=best_loss_model_path,
                    non_improved_epoch=non_improved_epoch,
                )

    wandb.finish()
    # os.system('shutdown')


if __name__ == "__main__":
    import os
    # sys.path.append("/home/zhaoming/MDA-RSM/model")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info("Interrupt")
        sys.exit(0)
