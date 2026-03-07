# 端到端车牌识别系统（YOLO + STN + CRNN）

本项目为毕业设计题目 **「端到端车牌识别系统」**，基于深度学习实现从车牌检测到车牌字符识别的一体化流程，特点是：

- 使用 **YOLOv8** 完成车牌区域检测；
- 使用 **STN + 注意力机制 + 去模糊模块 + CRNN** 进行车牌字符识别；
- 提供基于 `tkinter` 的 **可视化桌面界面**，并支持显示中间处理结果（STN 校正、去模糊、特征图等）；
- 提供 **完整训练代码**（YOLO 检测训练 + 端到端识别网络渐进式训练）。

> 适合作为车牌识别方向的课程设计 / 毕业设计示例项目。

## 1. 功能与特点

- **端到端流程**：从原始车辆图像输入，到最终车牌号码输出，无需手工裁剪车牌区域。
- **多模块融合**：
  - STN（Spatial Transformer Network）自动对车牌进行几何校正；
  - 去模糊模块提升模糊车牌的清晰度；
  - 注意力机制增强关键信息区域；
  - CRNN 序列网络完成字符级识别。
- **可视化友好**：
  - 图形界面展示检测框、车牌区域和识别结果；
  - Matplotlib 窗口展示 STN 校正结果、去模糊结果和特征图。
- **训练流程完整**：支持从 BLPD + CCPD 数据集出发，完成从检测到识别的训练与评估。

## 2. 项目结构

- `zhongduan.py`：推理 & GUI 程序，完成车牌检测 + 识别 + 中间结果可视化。
- `xunlianzonghe.py`：端到端识别网络训练脚本（渐进式学习 + 多任务学习）。
- `xunlianres2.py`：YOLOv8 车牌检测模型微调脚本。
- `models/`
  - `best.pt`：YOLOv8 车牌检测模型权重（需自行放入，不随仓库提供）。
  - `final_model.pth`：CRNN 识别模型权重（训练产物）。
  - `chars_mapping.json`：字符映射配置（训练脚本会生成一份，可复制到此处）。
- `runs/`：YOLO 训练产物（日志、结果等，可按需保留或忽略）。

## 3. 环境依赖

建议使用 **Python 3.10+**，并创建独立虚拟环境：

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

如有 GPU，确保已正确安装与 CUDA 匹配的 PyTorch 版本（参考 PyTorch 官方安装说明）。

## 4. 推理与界面使用

1. 在项目根目录下新建 `models` 目录，并放入：
   - `best.pt`：训练好的 YOLOv8 车牌检测模型；
   - `final_model.pth`：训练好的 CRNN 车牌识别模型；
   - `chars_mapping.json`：字符到索引的映射文件（可从训练输出目录复制）。
2. 激活虚拟环境，运行 GUI 程序：

   ```bash
   venv\Scripts\activate
   python zhongduan.py
   ```

3. 在界面中：
   - 点击「加载图像」选择包含车牌的图片；
   - 点击「识别车牌」：
     - 左侧：带检测框的原始图像；
     - 右侧：裁剪后的车牌区域；
     - 下方：识别出的车牌号码；
     - 同时弹出中间过程可视化窗口（STN 校正、去模糊、特征图等），便于论文/答辩展示模型工作原理。

## 5. 训练说明（简要）

### 5.1 YOLOv8 车牌检测训练（`xunlianres2.py`）

- 依赖已准备好的 CCPD 数据集以及对应的 `data.yaml` 配置；
- 在脚本中配置好数据集路径和预训练权重路径：
  - `dataset_path`
  - `model_path`
- 运行命令：

```bash
python xunlianres2.py
```

训练完成后，在配置的 `project` 目录中会生成 `best.pt` 等权重文件，可复制到 `models/best.pt` 供推理使用。

### 5.2 CRNN 端到端识别训练（`xunlianzonghe.py`）

- 使用 BLPD + CCPD 数据集，支持多阶段渐进式训练（基础 CRNN → 加 STN → 加去模糊 → 联合微调）；
- 在脚本顶部配置好：
  - `BLPD_DIR`、`BLPD_TRAIN_TXT`、`BLPD_VAL_TXT`
  - `CCPD_BASE_DIR` 等数据路径
  - `YOLO_MODEL_PATH`（用于裁剪车牌区域）
  - `OUTPUT_DIR`（输出目录，会保存日志、权重、可视化结果等）。

常用启动示例：

```bash
python xunlianzonghe.py --stage 1              # 从阶段1开始训练
python xunlianzonghe.py --resume               # 从上次中断处恢复训练
python xunlianzonghe.py --max-train-time 120   # 最长训练120分钟，到点自动安全暂停
```

训练结束后，`OUTPUT_DIR` 下会生成：

- `final_model.pth`：最终识别模型（推理时使用）；
- `chars_mapping.json`：字符映射文件；
- 日志文件、TensorBoard 日志、评估可视化图片等（可直接用于毕设报告中的结果展示）。

## 6. Git / GitHub 使用（简要）

如需把本目录单独作为一个 Git 仓库，可以在此目录下执行：

```bash
cd "D:\csgodemocache\end to end"

git init
git add .
git commit -m "初始化端到端车牌识别项目"
```

然后在 GitHub 新建仓库，将地址替换到下面命令中的 `YOUR_REPO_URL`：

```bash
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```

后续日常开发：

```bash
git status
git add .
git commit -m "本次修改的简要说明"
git push
```

