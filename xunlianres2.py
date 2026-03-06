from ultralytics import YOLO
import os
import torch.multiprocessing as mp

def main():
    # 首先检查数据集文件夹是否存在
    dataset_path = "E:/CCPD2019/CCPD2019/yolo_dataset"
    train_images_path = os.path.join(dataset_path, "train", "images")
    val_images_path = os.path.join(dataset_path, "val", "images")

    # 检查文件夹是否存在并包含图像
    print(f"训练集图像数量: {len(os.listdir(train_images_path))}")
    print(f"验证集图像数量: {len(os.listdir(val_images_path))}")

    # 加载已训练的模型
    model_path = "E:/CCPD2020/CCPD2020/ccpd_green/runs/detect/train/weights/best.pt"
    print(f"加载预训练模型: {model_path}")
    model = YOLO(model_path)

    # 设置训练参数
    print("开始训练...")
    results = model.train(
        data=dataset_path + "/data.yaml",
        epochs=50,               # 训练周期数
        batch=16,                # 批次大小
        imgsz=640,               # 图像尺寸
        device=0,                # 使用GPU
        patience=8,             # 早停参数
        save=True,               # 保存模型
        lr0=0.001,               # 起始学习率
        lrf=0.01,                # 最终学习率因子
        cache=True,              # 缓存图像以加速训练
        project="E:/CCPD2019/CCPD2019/YOLOv8_finetuned",  # 输出目录
        name="finetune_ccpd2019_fixed",  # 实验名称
        exist_ok=False,          # 不覆盖现有实验
        augment=True,            # 使用数据增强
        degrees=5.0,             # 旋转角度范围
        translate=0.1,           # 平移范围
        scale=0.5,               # 缩放范围
        fliplr=0.5,              # 水平翻转概率
        workers=0,               # 设置为0以避免Windows上的多进程问题
    )

    # 打印训练结果
    print(results)

    # 验证模型性能
    print("开始验证...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

    # 导出模型为ONNX格式，适合移动端部署
    print("导出模型为ONNX格式...")
    model.export(format="onnx", imgsz=640)
    print(f"模型已导出到: {model.export_dir}")

if __name__ == "__main__":
    # 在Windows上运行多进程程序需要这个保护
    mp.freeze_support()
    main()