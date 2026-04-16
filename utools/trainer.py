import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from net.dataset import CJJDataset
from net.loss_fun import MultiTaskLoss, ssim_loss
from net.swin_ir import SwinIR


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        lr=1e-4,
        device=None,
        save_path="best_model.pth"
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # CosineAnnealingWarmRestarts
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,       # 第一个周期（10个epoch）
            T_mult=2,     # 周期翻倍
            eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        self.save_path = save_path

    # =========================
    # 训练一个 epoch
    # =========================
    def train_one_epoch(self):

        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Train")

        for batch in pbar:


            inputs, targets = batch

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item(),
                             lr=self.optimizer.param_groups[0]['lr'])

        return total_loss / len(self.train_loader)

    # =========================
    # 验证
    # =========================
    def validate(self):

        self.model.eval()
        total_loss = 0
        total_ssim = 0

        with torch.no_grad():

            pbar = tqdm(self.val_loader, desc="Val")

            for batch in pbar:

                inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets)

                # =========================
                # SSIM
                # =========================
                ssim = ssim_loss(outputs, targets)

                if isinstance(ssim, torch.Tensor):
                    ssim = ssim.item()

                total_loss += loss.item()
                total_ssim += ssim

                pbar.set_postfix(
                    loss=loss.item(),
                    ssim=1-ssim
                )

        avg_loss = total_loss / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        return avg_loss, avg_ssim

    # =========================
    # 主训练循环
    # =========================
    def fit(self, epochs):

        for epoch in range(epochs):

            print(f"\nEpoch [{epoch+1}/{epochs}]")

            train_loss = self.train_one_epoch()
            val_loss, val_ssim = self.validate()
            ssim = 1 - val_ssim

            # 更新学习率（关键）
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val   Loss: {val_loss:.6f}")
            print(f"Val   SSIM: {ssim:.4f}")

            # =========================
            # 保存最佳模型
            # =========================
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss
                }, self.save_path)

                print("保存最佳模型")

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_config = {
        "Geo": "minmax",
        "OLR": "standard",
        "RH": "minmax",
        "UV_u": "standard",
        "UV_v": "standard",
        "tmax": "standard",
        "tmean": "standard",
        "tmin": "standard"
    }




    # =========================
    # 1️ 构建数据集
    # =========================
    dataset = CJJDataset(
        era5_root="../Data/ERA5/",
        cmfd_root="../Data/CMFD/",
        cma_root="../Data/CMA_CPSv3",
        era5_scaler="../Data/Scaler/ERA5.json",
        cmfd_scaler="../Data/Scaler/CMFD.json",
        cma_scaler="../Data/Scaler/CMA_CPSv3.json",
        scale_config=scale_config
    )

    # =========================
    # 2️ 按 7:2:1 划分
    # =========================
    total_size = len(dataset)

    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 固定随机种子（保证每次划分一致）
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print("总数:", total_size)
    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    print("test:", len(test_dataset))

    # =========================
    # 3️ 构建 DataLoader
    # =========================
    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    loss_fn = MultiTaskLoss()
    window_size = 4
    height = 32
    width = 52
    scale = 2.5
    model = SwinIR(upscale=scale,
                   img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=96, num_heads=[6, 6, 6, 6], mlp_ratio=4, resi_connection='3conv')

    ckpt = torch.load("../utools/0.8674.pth")

    model.load_state_dict(ckpt["model_state_dict"])
    # model.load_state_dict(torch.load("../utools/0.8674.pth"))
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        lr=1e-4,
        save_path="best_model.pth"
    )

    trainer.fit(epochs=50)