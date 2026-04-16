import json
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class CJJDataset(Dataset):
    def __init__(self, era5_root, cmfd_root,cma_root,
                 era5_scaler, cmfd_scaler,cma_scaler,
                 scale_config):

        self.era5_root = era5_root
        self.cmfd_root = cmfd_root
        self.cma_root = cma_root
        self.scale_config = scale_config

        # ===== 读取 scaler =====
        with open(era5_scaler, "r") as f:
            self.era5_scaler = json.load(f)

        with open(cma_scaler, "r") as f:
            self.cma_scaler = json.load(f)

        with open(cmfd_scaler, "r") as f:
            self.cmfd_scaler = json.load(f)



        # ===== 时间 =====
        geo_files = sorted(os.listdir(os.path.join(self.cma_root, "TEM/tmax")))
        self.times = [f.replace("tmax-", "") for f in geo_files]

        # ===== 变量配置 =====
        self.era5_vars = {
            "Geo": ("Geo", "geo-", False),
            "OLR": ("OLR", "olr-", False),
            "RH": ("RH", "rh-", False),
            "UV": ("UV", "uv-", True)
        }

        self.tem_vars = ["tmax", "tmean", "tmin"]

    # ===== 读取 tif =====
    def _read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)

    # ===== 核心 scaler（只听 config）=====
    def _apply_scaler(self, data, scaler, key):
        scale_type = self.scale_config[key]

        data = data.astype(np.float32)

        # ===== Standard =====
        if scale_type == "standard":

            mean = scaler["mean"]
            std = scaler["std"]
            return (data - mean) / (std + 1e-8)

        # ===== MinMax =====
        elif scale_type == "minmax":
            min_val = scaler["min"]
            max_val = scaler["max"]
            return (data - min_val) / (max_val - min_val + 1e-8)

        else:
            raise ValueError(f"Unknown scale type: {scale_type}")

    # ===== 加载变量 =====
    def _load_var(self, var_key, var_config, time_str, scaler_dict):
        folder, prefix, _ = var_config

        path = os.path.join(
            self.era5_root,
            folder,
            f"{prefix}{time_str}"
        )

        data = self._read_tif(path)

        # ===== UV（重点）=====
        if var_key == "UV":
            if data.shape[0] < 2:
                raise ValueError(f"UV不是双通道: {path}")

            u = data[0:1]
            v = data[1:2]
            u = self._apply_scaler(u, scaler_dict["UV_u"], "UV_u")
            v = self._apply_scaler(v, scaler_dict["UV_v"], "UV_v")

            return np.concatenate([u, v], axis=0)

        # ===== 普通变量 =====
        return self._apply_scaler(data, scaler_dict[var_key], var_key)

    # ===== TEM =====
    def _load_tem(self, root, time_str, scaler_dict):
        outputs = []

        for var in self.tem_vars:
            path = os.path.join(
                root,
                "TEM",
                var,
                f"{var}-{time_str}"
            )

            data = self._read_tif(path)
            data = self._apply_scaler(data, scaler_dict[var], var)

            outputs.append(data)

        return np.concatenate(outputs, axis=0)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time_str = self.times[idx]

        # ===== ERA5  and CMA_CPSv3 =====
        era5_list = []

        for var, config in self.era5_vars.items():
            data = self._load_var(var, config, time_str, self.era5_scaler)
            era5_list.append(data)

        # TEM 插入
        era5_tem = self._load_tem(
            self.era5_root,
            time_str,
            self.era5_scaler
        )

        era5_list.insert(3, era5_tem)

        cma_tem = self._load_tem(
            self.cma_root,
            time_str,
            self.cma_scaler
        )
        era5_list.insert(0, cma_tem)
        x = np.concatenate(era5_list, axis=0)


        # ===== CMFD =====
        cmfd_list = []

        for var in self.tem_vars:
            path = os.path.join(
                self.cmfd_root,
                "TEM",
                var,
                f"{var}-{time_str}"
            )

            data = self._read_tif(path)
            data = self._apply_scaler(data, self.cmfd_scaler[var], var)

            cmfd_list.append(data)

        y = np.concatenate(cmfd_list, axis=0)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

if __name__ == '__main__':
    from torch.utils.data import DataLoader


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
    dataset = CJJDataset(
        era5_root="../Data/ERA5/",
        cmfd_root="../Data/CMFD/",
        cma_root="../Data/CMA_CPSv3",
        era5_scaler="../Data/Scaler/ERA5.json",
        cmfd_scaler="../Data/Scaler/CMFD.json",
        cma_scaler="../Data/Scaler/CMA_CPSv3.json",
        scale_config=scale_config
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        # num_workers=2
    )

    # -------- 测试一个 batch --------
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        print(f"\nBatch {i}")

        print("x.shape:", x.shape)
        print("y.shape:", y.shape)

        # -------- 检查 NaN / Inf --------
        if torch.isnan(x).any():
            print("⚠️ x 存在 NaN")
        if torch.isinf(x).any():
            print("⚠️ x 存在 Inf")

        if torch.isnan(y).any():
            print("⚠️ y 存在 NaN")
        if torch.isinf(y).any():
            print("⚠️ y 存在 Inf")
        break