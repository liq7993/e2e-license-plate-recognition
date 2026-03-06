# 端到端车牌识别系统（YOLO + STN + CRNN）

本项目是一个 **端到端车牌识别系统**，包含：

- YOLOv8 车牌检测（含再训练脚本）
- STN + 注意力 + 去模糊 + CRNN 的车牌字符识别网络
- 基于 `tkinter` 的图形界面，可视化车牌检测与识别过程

## 1. 项目结构

- `zhongduan.py`：推理 & GUI 程序，完成车牌检测 + 识别 + 中间结果可视化。
- `xunlianzonghe.py`：端到端识别网络训练脚本（渐进式训练、多任务学习）。
- `xunlianres2.py`：YOLOv8 车牌检测模型微调脚本。
- `models/`
  - `best.pt`：YOLOv8 车牌检测模型权重（需自行放入，不随仓库提供）。
  - `final_model.pth`：CRNN 识别模型权重（训练产物）。
  - `chars_mapping.json`：字符映射配置。
- `runs/`：YOLO 训练产物（日志、结果，可选是否提交到仓库）。

## 2. 环境依赖

建议使用 **Python 3.10+**，并创建独立虚拟环境：

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

如有 GPU，确保已正确安装与 CUDA 匹配的 PyTorch 版本（可参考官网说明）。

## 3. 推理与界面使用

1. 在项目根目录下新建 `models` 目录，并放入：
   - `best.pt`：训练好的 YOLOv8 车牌检测模型。
   - `final_model.pth`：训练好的 CRNN 车牌识别模型。
   - `chars_mapping.json`：字符到索引的映射文件（训练脚本会自动生成一份，可复制到此处）。
2. 激活虚拟环境，进入项目目录：

   ```bash
   cd "D:\csgodemocache\复印件  毕设"
   venv\Scripts\activate
   python zhongduan.py
   ```

3. 在界面中：
   - 点击“加载图像”选择包含车牌的图片；
   - 点击“识别车牌”，界面会显示：
     - 左侧：检测框标注的原图；
     - 右侧：裁剪后的车牌区域；
     - 下方：识别的车牌号码；
     - 同时弹出中间过程可视化窗口（STN 校正、去模糊、特征图等）。

## 4. 训练说明（简要）

### 4.1 YOLOv8 车牌检测训练（`xunlianres2.py`）

- 依赖已准备好的 CCPD 数据集以及对应的 `data.yaml` 配置。
- 在脚本中配置好数据集路径和预训练权重路径：
  - `dataset_path`
  - `model_path`
- 直接运行：

```bash
python xunlianres2.py
```

训练完成后，在配置的 `project` 目录中会生成 `best.pt` 等权重文件，可复制到 `models/best.pt` 供推理使用。

### 4.2 CRNN 端到端识别训练（`xunlianzonghe.py`）

- 使用 BLPD + CCPD 数据集，支持多阶段渐进式训练（基础 CRNN → 加 STN → 加去模糊 → 联合微调）。
- 在脚本顶部配置好：
  - `BLPD_DIR`、`BLPD_TRAIN_TXT`、`BLPD_VAL_TXT`
  - `CCPD_BASE_DIR` 等数据路径
  - `YOLO_MODEL_PATH`（用于裁剪车牌区域）
  - `OUTPUT_DIR`（输出目录，会保存日志、权重、可视化结果等）

常用启动示例：

```bash
python xunlianzonghe.py --stage 1        # 从阶段1开始训练
python xunlianzonghe.py --resume         # 从上次中断处恢复训练
python xunlianzonghe.py --max-train-time 120  # 最长训练120分钟，到点自动安全暂停
```

训练结束后，`OUTPUT_DIR` 下会生成：

- `final_model.pth`：最终模型（推理时使用）。
- `chars_mapping.json`：字符映射文件。
- 日志文件、TensorBoard 日志、评估可视化图片等。

## 5. Git / GitHub 使用（简要）

在本地项目根目录（本文件所在目录）执行：

```bash
cd "D:\csgodemocache\复印件  毕设"

# 查看当前状态
git status

# 添加所有文件
git add .

# 首次提交
git commit -m "初始化端到端车牌识别毕设项目"
```

在 GitHub 上新建一个空仓库（例如 `e2e-license-plate-recognition`），然后在本地执行：

```bash
git branch -M main
git remote add origin https://github.com/你的用户名/e2e-license-plate-recognition.git
git push -u origin main
```

后续每次更新代码：

```bash
git add .
git commit -m "本次修改的简要说明"
git push
```

