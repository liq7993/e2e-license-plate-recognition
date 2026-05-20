# docs/screenshots/

放置项目演示截图与图表，用于 README 顶部展示。

建议放置的内容：

| 文件名 | 内容 |
|---|---|
| `gui_main.png` | GUI 主界面（加载图像 + 检测框 + 识别结果） |
| `gui_intermediate.png` | 中间结果窗（原始车牌 / STN 校正 / 去模糊 / 特征图） |
| `pipeline_overview.png` | 端到端识别一张图的全流程拼图 |
| `train_loss.png` | 4 阶段训练 loss 曲线（可选） |

放好后在 `README.md` 顶部加：

```markdown
![GUI 主界面](docs/screenshots/gui_main.png)
```

即可在 GitHub 首页直接渲染。

> 同样的隐私提醒：截图里出现的车牌请打码或使用 CCPD / BLPD 中的样本，不要泄露真实车辆信息。
