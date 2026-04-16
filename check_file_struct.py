import os


def print_tree(folder, indent=""):
    items = os.listdir(folder)

    dirs = [i for i in items if os.path.isdir(os.path.join(folder, i))]
    files = [i for i in items if os.path.isfile(os.path.join(folder, i))]

    # 打印文件夹
    for d in sorted(dirs):
        print(f"{indent}[D] {d}")
        print_tree(os.path.join(folder, d), indent + "    ")

    # 打印文件（最多2个示例）
    if files:
        print(f"{indent}[F] (showing 2 of {len(files)})")
        for f in sorted(files)[:2]:
            print(f"{indent}    {f}")


# ===== 修改这里 =====
root_path = r"Data/CMA_CPSv3/TEM"
# ====================

print(f"[ROOT] {root_path}")
print_tree(root_path)