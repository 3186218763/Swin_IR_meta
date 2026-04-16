import os
import json
import numpy as np
import rasterio
from collections import defaultdict

# ===== 修改路径 =====
root = r"Data/CMA_CPSv3"
save_path = r"Data/Scaler/CMA_CPSv3.json"
# ===================


def update_stats(stats, data):
    data = data.flatten()
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return

    stats["count"] += len(data)
    stats["sum"] += data.sum()
    stats["sum2"] += (data ** 2).sum()
    stats["min"] = min(stats["min"], data.min())
    stats["max"] = max(stats["max"], data.max())


def finalize_stats(stats):
    mean = stats["sum"] / stats["count"]
    std = np.sqrt(stats["sum2"] / stats["count"] - mean ** 2)
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(stats["min"]),
        "max": float(stats["max"])
    }


# 初始化
stats_dict = defaultdict(lambda: {
    "count": 0,
    "sum": 0.0,
    "sum2": 0.0,
    "min": float("inf"),
    "max": float("-inf")
})


def process_folder(folder, key):
    for file in os.listdir(folder):
        if file.endswith(".tif"):
            path = os.path.join(folder, file)
            with rasterio.open(path) as src:
                data = src.read(1)
                update_stats(stats_dict[key], data)


# ===== 处理 =====
#
# # Geo（可选）
# process_folder(os.path.join(root, "Geo"), "Geo")
#
# # OLR
# process_folder(os.path.join(root, "OLR"), "OLR")
#
# # RH
# process_folder(os.path.join(root, "RH"), "RH")
#
# # ===== UV（重点）=====
# uv_folder = os.path.join(root, "UV")
#
# for file in os.listdir(uv_folder):
#     if file.endswith(".tif"):
#         path = os.path.join(uv_folder, file)
#         with rasterio.open(path) as src:
#
#             # ✅ 强制检查通道数
#             if src.count < 2:
#                 print(f"⚠️ 跳过（不是双通道）: {file}")
#                 continue
#
#             u = src.read(1)  # U分量
#             v = src.read(2)  # V分量
#
#             update_stats(stats_dict["UV_u"], u)
#             update_stats(stats_dict["UV_v"], v)


# ===== TEM =====
tem_root = os.path.join(root, "TEM")
for sub in ["tmax", "tmean", "tmin"]:
    process_folder(os.path.join(tem_root, sub), sub)


# ===== 汇总 =====
final_dict = {}
for key in stats_dict:
    final_dict[key] = finalize_stats(stats_dict[key])


# ===== 保存 =====
with open(save_path, "w") as f:
    json.dump(final_dict, f, indent=4)

print("✅ Scaler 已保存:", save_path)