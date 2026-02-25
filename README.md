# 面向脑肿瘤辅助诊断的 MRI 图像语义分割系统（BraTS2019）

本项目用于复现开题报告《面向脑肿瘤辅助诊断的MRI图像语义分割研究与系统实现》的核心技术路线，包含：
- 多模态 MRI 3D 语义分割训练与评估
- 主模型（改进 3D U-Net）与对比模型（V-Net / UNETR）
- 全体积滑窗推理
- FastAPI 推理服务 + Web 前端
- PDF 报告与中间产物导出
- 训练/评估/系统日志体系

Windows 用户可直接参考专用训练文档：
- `docs/Windows_Training_Guide.md`

---

## 1. 项目目标

### 1.1 研究目标
本项目面向脑肿瘤辅助诊断场景，基于 BraTS2019 多模态 MRI 数据，实现自动化肿瘤分割，并支持系统化部署。

### 1.2 任务定义
- 输入：每例 4 个模态 MRI（`t1`/`t1ce`/`t2`/`flair`）
- 输出：体素级语义分割标签（BraTS 标注体系）
- 训练内部标签映射：`{0,1,2,4} -> {0,1,2,3}`，推理输出再映射回 `{0,1,2,4}`

### 1.3 评估指标
- Dice（DSC）
- IoU
- SEN（灵敏度）
- SPE（特异度）

---

## 2. 数据集与目录规范

### 2.1 数据集
使用本地数据集：
- `data/MICCAI_BraTS_2019_Data_Training`

数据结构：
- `HGG/`、`LGG/` 两类病例目录
- 每个病例包含：
  - `*_flair.nii`
  - `*_t1.nii`
  - `*_t1ce.nii`
  - `*_t2.nii`
  - `*_seg.nii`

### 2.2 数据划分
按开题报告使用 `7:2:1`（训练/验证/测试）划分：
```bash
python3 scripts/generate_split.py \
  --data-root data/MICCAI_BraTS_2019_Data_Training \
  --output artifacts/splits/split_70_20_10.json \
  --seed 42
```

---

## 3. 技术方案与实现对应

### 3.1 预处理（与开题报告对齐）
实现位置：`brats_seg/data/preprocess.py`, `brats_seg/data/dataset.py`

- 模态独立标准化：`Z-score`（非零区域）
- 去噪：`3x3x3` 中值滤波 + `sigma=1.0` 高斯滤波
- N4 强度非均匀校正：
  - 默认启用
  - 以 `t1ce` 非零区域生成脑组织掩码
  - 收敛阈值 `1e-6`
- 标签重映射：`4 -> 3`（训练内部）

### 3.2 数据增强
实现位置：`brats_seg/data/augment.py`

- 随机翻转
- 随机旋转（`±10°`）
- 随机缩放（`0.9~1.1`）
- GridSample 弹性形变（形变系数 `0.05`）

### 3.3 主模型：改进 3D U-Net
实现位置：`brats_seg/models/improved_unet3d.py`, `brats_seg/models/blocks.py`

核心改进：
- 第2层多模态融合（四模态特征融合）
- 跳跃连接注意力门控（逐元素相乘路径）
- 最终 `1x1x1` 输出 4 类分割结果

### 3.4 对比模型
实现位置：`brats_seg/models/factory.py`, `brats_seg/models/reported_vnet.py`

- V-Net：报告对齐通道结构（`16->32->64->128->256`，PReLU），训练使用 Dice 单损失，输入 patch 固定 `128x128x128`
- UNETR：12 heads，hidden=768；按序列长度 1024 设置输入尺寸策略

### 3.5 训练与评估策略
实现位置：`train.py`, `evaluate.py`, `brats_seg/training/*`

- 优化器：AdamW
- 学习率衰减：每 10 epoch 乘 0.9
- 早停：验证集连续 5 epoch 无提升
- 损失：Dice + CE
- 验证/评估：全体积滑窗推理（默认 patch `128^3`，overlap=0.5）
- UNETR：默认自动调整 patch 为 `128x128x256`

---

## 4. 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 建议 Python 3.9+，并安装可用的 CUDA/PyTorch（若使用 GPU）。

---

## 5. 训练、评估与对比实验

### 5.1 训练主模型
```bash
python3 train.py \
  --model improved_unet3d \
  --split-json artifacts/splits/split_70_20_10.json \
  --out-dir artifacts/exp_main \
  --epochs 100 \
  --batch-size 1 \
  --patch-size 128 128 128
```

### 5.2 训练对比模型
```bash
python3 train.py --model vnet  --out-dir artifacts/exp_vnet
python3 train.py --model unetr --out-dir artifacts/exp_unetr
```

### 5.3 单模型评估
```bash
python3 evaluate.py \
  --model improved_unet3d \
  --split-json artifacts/splits/split_70_20_10.json \
  --ckpt artifacts/exp_main/best_model.pt \
  --out-dir artifacts/eval_main
```

### 5.4 对比实验汇总（论文表格）
```bash
python3 scripts/compare_experiments.py \
  --split-json artifacts/splits/split_70_20_10.json \
  --out-dir artifacts/reports
```

输出：
- `comparison.json`
- `comparison.csv`
- `comparison.md`

---

## 6. ONNX/TensorRT 部署链路

### 6.1 导出 ONNX
```bash
python3 scripts/export_onnx.py \
  --model improved_unet3d \
  --ckpt artifacts/exp_main/best_model.pt \
  --output artifacts/onnx/improved_unet3d.onnx
```

### 6.2 构建 TensorRT Engine（可选）
```bash
python3 scripts/build_trt_engine.py \
  --onnx artifacts/onnx/improved_unet3d.onnx \
  --engine artifacts/trt/improved_unet3d_fp16.engine
```

---

## 7. 系统服务与前端

### 7.1 启动服务
```bash
export BRATS_API_TOKEN='your-secure-token'
export MODEL_NAME='improved_unet3d'   # or vnet / unetr
export MODEL_CKPT='artifacts/exp_main/best_model.pt'
export ONNX_PATH='artifacts/onnx/improved_unet3d.onnx'
export TRT_ENGINE_PATH='artifacts/trt/improved_unet3d_fp16.engine'
export INFER_BACKEND='pytorch'        # pytorch / onnx / tensorrt

python3 -m system.run_server
```

### 7.2 访问入口
- 前端：`http://127.0.0.1:8000/ui`
- 健康检查：`http://127.0.0.1:8000/api/health`

### 7.3 API 概览
- `POST /api/tasks`：上传 4 模态并创建任务
- `GET /api/tasks/{task_id}`：查询状态
- `GET /api/tasks/{task_id}/preview/{view}`：预览图
- `GET /api/tasks/{task_id}/seg`：下载分割 NIfTI
- `GET /api/tasks/{task_id}/report`：下载 PDF 报告

认证方式：
- Header：`x-api-token: <token>`
- Query：`?token=<token>`（用于下载链接）

---

## 8. 日志系统与论文产物

### 8.1 训练日志与产物
每次训练会在 `artifacts/exp_xxx/` 生成：
- `logs/train.log`：详细训练日志
- `logs/metrics.jsonl`：逐 epoch 指标流
- `history.json` / `history.csv`：训练历程
- `run_config.json`：实验配置快照
- `summary.json`：最佳结果摘要
- `intermediate/npz/*.npz`：中间样本（`image/label/pred/logits`）
- `intermediate/figures/*_gt.png`、`*_pred.png`：中间可视化

训练参数（控制日志与中间样本）：
```bash
python3 train.py \
  --out-dir artifacts/exp_main \
  --log-level INFO \
  --save-val-samples 2 \
  --save-val-every 1
```

### 8.2 评估日志
`evaluate.py` 输出：
- `artifacts/eval_xxx/eval.log`
- `artifacts/eval_xxx/metrics.json`

### 8.3 后端系统日志
系统运行期间：
- `system/storage/tasks/backend.log`：服务日志
- `system/storage/tasks/audit.log`：任务审计轨迹

---

## 9. 端到端自测（Smoke Test）

服务启动后执行：
```bash
python3 scripts/e2e_smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --token "$BRATS_API_TOKEN" \
  --backend pytorch \
  --data-root data/MICCAI_BraTS_2019_Data_Training
```

脚本会自动：
- 上传 4 模态
- 轮询任务完成
- 下载 NIfTI 与 PDF 并做非空校验

---

## 10. 项目结构说明

```text
brats_seg/
  data/              # 数据索引、预处理、增强、Dataset
  models/            # 主模型与对比模型
  training/          # 训练/验证引擎、推理、损失、指标、产物导出
  utils.py           # 通用工具（日志、json/csv）

system/
  backend/           # FastAPI后端、推理、任务管理、报告、日志
  frontend/          # Web界面
  run_server.py      # 服务启动入口

scripts/
  generate_split.py      # 数据划分
  export_onnx.py         # ONNX导出
  build_trt_engine.py    # TensorRT引擎构建
  compare_experiments.py # 对比实验汇总
  e2e_smoke_test.py      # 端到端烟雾测试

train.py
evaluate.py
```

---

## 11. 常见问题

1. `SimpleITK` 未安装导致 N4 报错
- 安装：`pip install SimpleITK`

2. UNETR 输入尺寸不匹配
- 脚本已自动将默认 patch 调整到 `128x128x256`

3. ONNX/TensorRT 不可用
- 检查 `ONNX_PATH` / `TRT_ENGINE_PATH` 文件存在
- TensorRT 需要 `trtexec`、`tensorrt`、`pycuda`

4. 显存不足
- 减小 `patch-size`
- 降低 `batch-size`
- 减少 `num-workers`

---

## 12. 隐私与数据安全

- 上传文件会以任务 ID 重命名，避免直接暴露原始命名
- 本地目录隔离存储：`uploads/results/reports/tasks`
- 支持离线部署，不依赖外网

> 说明：本仓库默认开发环境配置，生产环境可按需求追加 HTTPS、反向代理、细粒度 RBAC。
