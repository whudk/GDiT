import torch


def get_state_dict(ckpt,name = 'model'):
    """根据常见格式，从 ckpt 中取出 state_dict."""
    # 如果是直接就是 state_dict


    return ckpt[name]  # 实在不行就原样返回


# 1. 加载
ckpt = torch.load(r"F:\M2I\comparision\0035000_best.pt", map_location="cpu")
tea_ckpt = torch.load(r"E:\GDiT\MaskDiT-master\results\DiT-XL\2-fusion_type-AdaLn\checkpoints\regions_graph_sem.pt", map_location="cpu")

sd = get_state_dict(ckpt,'model')
tea_sd = get_state_dict(tea_ckpt,'teacher')

# 2. key 层面比较
keys_sd = set(sd.keys())
keys_tea = set(tea_sd.keys())

only_in_sd = sorted(list(keys_sd - keys_tea))
only_in_tea = sorted(list(keys_tea - keys_sd))
common_keys = sorted(list(keys_sd & keys_tea))

print("=== 仅在 ckpt 中存在的参数 ===")
for k in only_in_sd:
    print("  ", k)

print("\n=== 仅在 tea_ckpt 中存在的参数 ===")
for k in only_in_tea:
    print("  ", k)

# 3. 对共同的 key 比较 shape 和数值差异
print("\n=== 共同参数的 shape / 数值差异 ===")
diff_shapes = []
diff_values = []

for k in common_keys:
    v1 = sd[k]
    v2 = tea_sd[k]

    if v1.shape != v2.shape:
        diff_shapes.append((k, v1.shape, v2.shape))
        continue

    # 形状一样再比较数值
    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        # max |v1 - v2|
        diff = (v1 - v2).abs()
        max_diff = diff.max().item()
        # 也可以根据需要设一个阈值
        if max_diff > 0:
            diff_values.append((k, max_diff))

print(">>> shape 不一致的参数:")
for k, s1, s2 in diff_shapes:
    print(f"  {k}: ckpt={s1}, tea_ckpt={s2}")

print("\n>>> 数值不完全相同的参数(显示前 50 个):")
for k, max_diff in diff_values[:50]:
    print(f"  {k}: max |Δ| = {max_diff:.6f}")

print(f"\n共有参数数目: {len(common_keys)}")
print(f"shape 不同的参数数目: {len(diff_shapes)}")
print(f"数值有差异的参数数目: {len(diff_values)}")
