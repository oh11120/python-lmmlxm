# Windows 训练指南（主模型 + 对比模型）

本文档面向 Windows 用户，详细说明如何在本项目中完成：
- 主模型（`improved_unet3d`）训练
- 对比模型（`vnet`、`unetr`）训练
- 训练日志查看
- 中间文件导出（用于论文）
- 评估与对比结果汇总

> 默认使用 **PowerShell**。如果使用 CMD，请将环境变量和换行方式改为 CMD 语法。

---

## 1. 前置条件

### 1.1 系统和硬件建议
- Windows 10/11（64位）
- NVIDIA GPU（建议显存 >= 12GB）
- NVIDIA 驱动已正确安装

### 1.2 软件建议版本
- Python: 3.9 ~ 3.11
- Git: 最新稳定版
- CUDA: 与已安装 PyTorch 版本匹配

---

## 2. 打开项目

在 PowerShell 中进入项目目录：
```powershell
cd D:\your_path\python-llmlxm
```

确认数据目录存在：
```powershell
Get-ChildItem .\data\MICCAI_BraTS_2019_Data_Training
```

你应能看到 `HGG`、`LGG` 等目录。

---

## 3. 创建并激活虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查 Python 和 PyTorch：
```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 4. 生成数据划分（7:2:1）

```powershell
python .\scripts\generate_split.py `
  --data-root .\data\MICCAI_BraTS_2019_Data_Training `
  --output .\artifacts\splits\split_70_20_10.json `
  --seed 42
```

成功后会生成：
- `artifacts/splits/split_70_20_10.json`

---

## 5. 主模型训练（improved_unet3d）

## 5.1 推荐命令（包含详细日志和中间产物）
```powershell
python .\train.py `
  --model improved_unet3d `
  --split-json .\artifacts\splits\split_70_20_10.json `
  --out-dir .\artifacts\exp_main `
  --epochs 100 `
  --batch-size 1 `
  --patch-size 128 128 128 `
  --num-workers 2 `
  --log-level INFO `
  --save-val-samples 2 `
  --save-val-every 1
```

## 5.2 训练过程会生成什么
目录：`artifacts/exp_main/`
- `best_model.pt`：最佳模型（按验证 Dice）
- `last_model.pt`：最后一轮模型
- `logs/train.log`：详细训练日志
- `logs/metrics.jsonl`：逐 epoch 指标流
- `history.csv`：表格化训练曲线（论文可直接用）
- `history.json`：训练历史
- `run_config.json`：本次实验配置快照
- `summary.json`：最佳指标摘要
- `intermediate/npz/*.npz`：中间样本（image/label/pred/logits）
- `intermediate/figures/*_gt.png`、`*_pred.png`：中间可视化图

## 5.3 实时查看日志
```powershell
Get-Content .\artifacts\exp_main\logs\train.log -Wait
```

---

## 6. 对比模型训练

## 6.1 V-Net 训练
```powershell
python .\train.py `
  --model vnet `
  --split-json .\artifacts\splits\split_70_20_10.json `
  --out-dir .\artifacts\exp_vnet `
  --epochs 100 `
  --batch-size 1 `
  --patch-size 128 128 128 `
  --num-workers 2 `
  --log-level INFO `
  --save-val-samples 1 `
  --save-val-every 2
```

说明：
- `vnet` 会自动使用 Dice 单损失（与开题报告一致）
- `vnet` 的输入 patch 会自动固定为 `128x128x128`

## 6.2 UNETR 训练
```powershell
python .\train.py `
  --model unetr `
  --split-json .\artifacts\splits\split_70_20_10.json `
  --out-dir .\artifacts\exp_unetr `
  --epochs 100 `
  --batch-size 1 `
  --num-workers 2 `
  --log-level INFO `
  --save-val-samples 1 `
  --save-val-every 2
```

说明：
- `unetr` 会自动将默认 patch 调整为 `128x128x256`（序列长度 1024 约束）。

---

## 7. 单模型评估

## 7.1 评估主模型
```powershell
python .\evaluate.py `
  --model improved_unet3d `
  --split-json .\artifacts\splits\split_70_20_10.json `
  --ckpt .\artifacts\exp_main\best_model.pt `
  --out-dir .\artifacts\eval_main `
  --num-workers 2
```

## 7.2 评估输出
- `artifacts/eval_main/eval.log`
- `artifacts/eval_main/metrics.json`

---

## 8. 汇总主模型与对比模型结果（论文表格）

```powershell
python .\scripts\compare_experiments.py `
  --split-json .\artifacts\splits\split_70_20_10.json `
  --out-dir .\artifacts\reports
```

输出文件：
- `artifacts/reports/comparison.json`
- `artifacts/reports/comparison.csv`
- `artifacts/reports/comparison.md`

你可以直接将 `comparison.csv` 导入 Excel 或论文表格模板。

---

## 9. 推荐训练组织方式（论文写作友好）

建议每次实验单独目录：
- `artifacts/exp_main_seed42`
- `artifacts/exp_vnet_seed42`
- `artifacts/exp_unetr_seed42`

示例：
```powershell
python .\train.py --model improved_unet3d --out-dir .\artifacts\exp_main_seed42 --seed 42
python .\train.py --model improved_unet3d --out-dir .\artifacts\exp_main_seed3407 --seed 3407
```

这样可以在论文中报告不同随机种子的稳定性结果。

---

## 10. 常见问题与处理

## 10.1 CUDA 不可用
现象：`torch.cuda.is_available() == False`
- 检查 NVIDIA 驱动
- 安装与 CUDA 匹配的 PyTorch
- 先用 CPU 跑通：降低 `num-workers`，减少 `epochs` 验证流程

## 10.2 显存不足（CUDA out of memory）
- 降低 `batch-size`（保持 1）
- 减少 patch 大小（非 UNETR 情况可尝试）
- 减少并行进程：`--num-workers 0`

## 10.3 数据读取慢
- 数据放在 SSD
- `--num-workers` 从 0/2/4 逐步试
- 避免同时运行多训练任务占用磁盘

## 10.4 N4 相关错误
N4 默认启用。若环境问题临时排查可先关闭：
```powershell
python .\train.py --disable-n4
```

---

## 11. 一键最小流程（先跑通）

如果你想快速验证全链路：

```powershell
python .\scripts\generate_split.py --data-root .\data\MICCAI_BraTS_2019_Data_Training --output .\artifacts\splits\split_70_20_10.json --seed 42
python .\train.py --model improved_unet3d --split-json .\artifacts\splits\split_70_20_10.json --out-dir .\artifacts\exp_main_quick --epochs 2 --batch-size 1 --num-workers 0 --save-val-samples 1
python .\evaluate.py --model improved_unet3d --split-json .\artifacts\splits\split_70_20_10.json --ckpt .\artifacts\exp_main_quick\best_model.pt --out-dir .\artifacts\eval_main_quick --num-workers 0
```

跑通后再把 `epochs` 调回正式实验配置。

---

## 12. 建议提交到论文附录的文件

建议至少保留并归档以下文件：
- `run_config.json`
- `summary.json`
- `history.csv`
- `comparison.csv`
- `intermediate/figures/` 中每轮代表图
- 关键日志：`logs/train.log`, `eval.log`

这些文件可以完整支撑你在论文中描述“参数设置、训练过程、结果对比和可解释可视化”。
