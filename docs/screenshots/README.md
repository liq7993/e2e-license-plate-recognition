# docs/screenshots/

放置项目演示截图与图表，用于 README 顶部展示。

## 已有

| 文件 | 来源 | 重新生成方式 |
|---|---|---|
| `training_overview.png` | 由 `_gen_training_overview.py` 从 `runs/detect/train/results.csv` 生成的 2×2 训练总览（train/val loss + 指标曲线 + 终态指标卡） | `python docs/screenshots/_gen_training_overview.py` |

如果未来 `runs/detect/train/results.csv` 更新了，直接重跑脚本即可覆盖图。

## 还建议补的（需要本地跑一遍才能截图）

| 文件名 | 内容 |
|---|---|
| `gui_main.png` | GUI 主界面（加载图像 + 检测框 + 识别结果） |
| `gui_intermediate.png` | 中间结果窗（原始车牌 / STN 校正 / 去模糊 / 特征图） |
| `pipeline_overview.png` | 端到端识别一张图的全流程拼图 |

放好后在 `README.md` 顶部加：

```markdown
![GUI 主界面](docs/screenshots/gui_main.png)
```

即可在 GitHub 首页直接渲染。

> 隐私提醒：截图里出现的车牌请打码或使用 CCPD / BLPD 中的样本，
> 不要把含真实车牌的非授权照片提交到本仓库。
