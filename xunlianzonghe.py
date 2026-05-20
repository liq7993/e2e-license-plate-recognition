#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端车牌识别 —— 4 阶段渐进式训练脚本
=======================================

在 CCPD2019 / CCPD2020 + BLPD 数据集上训练 EnhancedCRNN：
    STN  +  MobileNetV3-Small  +  Channel/Spatial Attention
         +  Deblur 辅助分支    +  BiLSTM + CTC

通过分阶段引入难任务（基础 CRNN → +STN → +Deblur → 联合微调），
避免一次性堆叠模块导致训练不稳。

用法
----
    # 完整 4 阶段顺序训练
    python xunlianzonghe.py

    # 从第 3 阶段开始
    python xunlianzonghe.py --stage 3

    # 从最近的检查点恢复
    python xunlianzonghe.py --resume

    # 限制最长训练时间（分钟）
    python xunlianzonghe.py --max-train-time 480

    # 设定每天自动暂停时间
    python xunlianzonghe.py --pause-time 23:00

也可以创建 pause_training.txt 文件，让训练在下一个 epoch 边界安全停止。

路径配置
--------
全部通过 **环境变量** 覆盖，无需改源码：

    BLPD_DIR / BLPD_TRAIN_TXT / BLPD_VAL_TXT
    CCPD_BASE_DIR
    YOLO_MODEL_PATH      # 用于在 CCPD 上裁车牌的 YOLOv8 权重
    OUTPUT_DIR           # 训练产物 / 日志 / 检查点输出目录

详见仓库 README 中的「环境变量」对照表。

输出
----
- stageX_best_model.pth        每阶段最佳权重
- stageX_checkpoint_epoch_K.pth 中断恢复用
- chars_mapping.json            字符 ↔ 索引映射（推理时必需）
- logs/                         TensorBoard 日志
- end_to_end_lpr.log            训练日志
"""

import os
import sys
import time
import datetime
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
import random
import math
import logging
import traceback
import re
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from collections import OrderedDict
from skimage.draw import line
from ultralytics import YOLO
import torch.multiprocessing as mp

# 命令行参数解析
parser = argparse.ArgumentParser(description='端到端车牌识别系统训练')
parser.add_argument('--pause-time', type=str, default='23:00', 
                    help='训练自动暂停时间 (格式: HH:MM)')
parser.add_argument('--resume', action='store_true', 
                    help='从上次保存的检查点恢复训练')
parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3, 4],
                    help='指定从哪个阶段开始训练 (1-4)')
parser.add_argument('--max-train-time', type=int, default=0,
                    help='最大训练时间(分钟)，0表示不限制')
parser.add_argument('--pause-file', type=str, default='pause_training.txt',
                    help='如果此文件存在，将暂停训练')
args = parser.parse_args()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("end_to_end_lpr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("脚本开始执行")
logger.info(f"命令行参数: {args}")

# ======== 路径配置 ========
# 以下路径均可通过同名环境变量覆盖，避免改源码即可在不同机器上跑：
#   BLPD_DIR / BLPD_TRAIN_TXT / BLPD_VAL_TXT
#   CCPD_BASE_DIR
#   YOLO_MODEL_PATH
#   OUTPUT_DIR
# 例如 (Windows PowerShell)：
#   $env:BLPD_DIR="D:\datasets\BLPD"; $env:CCPD_BASE_DIR="D:\datasets\CCPD2019"
#   python xunlianzonghe.py
# Linux / macOS：
#   BLPD_DIR=/data/BLPD CCPD_BASE_DIR=/data/CCPD2019 python xunlianzonghe.py

# 原始BLPD数据集路径
BLPD_DIR = os.environ.get("BLPD_DIR", r"E:\BLPD")
BLPD_TRAIN_TXT = os.environ.get("BLPD_TRAIN_TXT", os.path.join(BLPD_DIR, "train.txt"))
BLPD_VAL_TXT = os.environ.get("BLPD_VAL_TXT", os.path.join(BLPD_DIR, "val.txt"))

# CCPD2019数据集路径
CCPD_BASE_DIR = os.environ.get("CCPD_BASE_DIR", r"E:\CCPD2019\CCPD2019")
CCPD_BLUR_DIR = os.path.join(CCPD_BASE_DIR, "blur")
CCPD_WEATHER_DIR = os.path.join(CCPD_BASE_DIR, "weather")
CCPD_TILT_DIR = os.path.join(CCPD_BASE_DIR, "tilt")

# YOLOv8训练好的模型路径
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    r"E:\CCPD2019\CCPD2019\YOLOv8_finetuned\finetune_ccpd2019_fixed\weights\best.pt",
)

# 输出目录
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"C:\Users\32044\Desktop\xunlianrcnn")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"输出目录: {OUTPUT_DIR}")

# 设置暂停和恢复文件路径
PAUSE_FILE = os.path.join(OUTPUT_DIR, args.pause_file)
RESUME_CONFIG_FILE = os.path.join(OUTPUT_DIR, 'resume_config.json')

# ======== 模型参数 ========
# 阶段标志位
ENABLE_STN = True      # 是否启用空间变换器
ENABLE_DEBLUR = True   # 是否启用去模糊
ENABLE_ATTENTION = True  # 是否使用注意力机制
USE_MOBILENET = True   # 是否使用MobileNetV3（否则使用ResNet18）

# 渐进式学习的阶段设置
STAGE1_EPOCHS = 15  # 基础CRNN训练
STAGE2_EPOCHS = 10  # 添加STN
STAGE3_EPOCHS = 10  # 添加去模糊任务
STAGE4_EPOCHS = 10  # 联合微调

# 渐进式学习阶段名称列表
STAGE_NAMES = ['stage1', 'stage2', 'stage3', 'stage4']

# 多任务损失权重
RECOGNITION_WEIGHT = 1.0  # 识别任务权重
DEBLUR_WEIGHT_INITIAL = 0.1  # 去模糊任务初始权重
DEBLUR_WEIGHT_FINAL = 0.5  # 去模糊任务最终权重

# 训练参数设置
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
IMAGE_HEIGHT = 32  # 车牌图像的标准高度
IMAGE_WIDTH = 160  # 车牌图像的标准宽度
NUM_WORKERS = 4

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    logger.info(f"CUDA版本: {torch.version.cuda}")
    logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# ======== 字符集定义 ========
# 中国省份简称 + 字母 + 数字
PROVINCES = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽", "赣", 
             "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", 
             "青", "宁", "新", "港", "澳", "台"]
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z')+1)]
DIGITS = [str(i) for i in range(10)]
CHARS = PROVINCES + ALPHABETS + DIGITS

logger.info(f"字符集大小: {len(CHARS)}")
# 检查是否有重复字符
if len(CHARS) != len(set(CHARS)):
    logger.warning("字符集中存在重复字符!")
    for char in CHARS:
        if CHARS.count(char) > 1:
            logger.warning(f"重复字符: {char}")

# 特殊字符，用于CTC处理
BLANK_CHAR = '-'  # CTC blank字符
CHARS_MAP = {char: i+1 for i, char in enumerate(CHARS)}  # 0保留给blank
IDX_TO_CHARS = {i+1: char for i, char in enumerate(CHARS)}
IDX_TO_CHARS[0] = BLANK_CHAR
NUM_CLASSES = len(CHARS) + 1  # +1 for blank

# 创建Tensorboard写入器
writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'logs'))

# ======== 暂停和恢复函数 ========
def should_pause_training():
    """检查是否应该暂停训练"""
    # 检查暂停文件是否存在
    if os.path.exists(PAUSE_FILE):
        logger.info(f"发现暂停文件: {PAUSE_FILE}，将暂停训练")
        return True
    
    # 检查是否到达指定暂停时间
    if args.pause_time:
        try:
            pause_hour, pause_minute = map(int, args.pause_time.split(':'))
            now = datetime.datetime.now()
            if now.hour == pause_hour and now.minute >= pause_minute:
                logger.info(f"到达指定的暂停时间: {args.pause_time}，将暂停训练")
                return True
        except Exception as e:
            logger.error(f"解析暂停时间出错: {e}")
    
    # 检查是否超过最大训练时间
    if args.max_train_time > 0:
        if 'global_start_time' in globals():
            elapsed_minutes = (time.time() - global_start_time) / 60
            if elapsed_minutes >= args.max_train_time:
                logger.info(f"已达到最大训练时间: {args.max_train_time}分钟，将暂停训练")
                return True
    
    return False

def save_resume_config(stage, epoch, best_model_info=None):
    """保存恢复配置"""
    config = {
        'stage': stage,
        'epoch': epoch,
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model_path': os.path.join(OUTPUT_DIR, f'{STAGE_NAMES[stage-1]}_best_model.pth') if best_model_info else None,
        'checkpoint_path': os.path.join(OUTPUT_DIR, f'{STAGE_NAMES[stage-1]}_checkpoint_epoch_{epoch}.pth')
    }
    
    # 保存配置到JSON文件
    with open(RESUME_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"保存恢复配置到: {RESUME_CONFIG_FILE}")
    return config

def load_resume_config():
    """加载恢复配置"""
    if os.path.exists(RESUME_CONFIG_FILE):
        try:
            with open(RESUME_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"加载恢复配置: {config}")
            return config
        except Exception as e:
            logger.error(f"加载恢复配置出错: {e}")
    
    logger.warning(f"恢复配置文件不存在: {RESUME_CONFIG_FILE}")
    return None

def clean_pause_file():
    """清除暂停文件"""
    if os.path.exists(PAUSE_FILE):
        try:
            os.remove(PAUSE_FILE)
            logger.info(f"已清除暂停文件: {PAUSE_FILE}")
        except Exception as e:
            logger.error(f"清除暂停文件出错: {e}")

# ======== CCPD2019文件名解析 ========
def parse_ccpd_filename(filename):
    """
    解析CCPD2019数据集的文件名格式
    
    格式: area_tilt_carplate_location_vertices_models_brightness_blurriness_weather
    
    例如: 01-1_3-263&456_407&514-407&510_268&514_263&460_402&456-0_0_10_23_32_28_33-166-2
    
    返回: 解析后的车牌号码
    """
    try:
        # 分离文件名部分
        parts = filename.split('-')
        
        # 车牌部分在第三部分
        plate_part = parts[2]
        
        # CCPD的车牌号码编码: 第一位是省份，后面是字母+数字
        plate_indexes = plate_part.split('_')
        
        # 解析省份 (PROVINCES索引)
        province_idx = int(plate_indexes[0])
        province = PROVINCES[province_idx % len(PROVINCES)]
        
        # 解析字母+数字 (分别是ALPHABETS和DIGITS索引)
        other_chars = []
        for char_index in plate_indexes[1].split('&'):
            idx = int(char_index)
            # 如果索引小于字母表长度，则是字母
            if idx < len(ALPHABETS):
                other_chars.append(ALPHABETS[idx])
            # 否则是数字，需要减去字母表长度
            else:
                idx -= len(ALPHABETS)
                if idx < len(DIGITS):
                    other_chars.append(DIGITS[idx])
        
        # 组合成完整车牌号
        plate_number = province + ''.join(other_chars)
        return plate_number
    except Exception as e:
        logger.error(f"解析CCPD文件名时出错: {e}, 文件名: {filename}")
        return None

# ======== 数据集类 ========
class CombinedLicensePlateDataset(Dataset):
    def __init__(self, blpd_txt_path=None, ccpd_dirs=None, transform=None, augment=False, 
                 use_yolo=False, yolo_model=None, phase='train'):
        """
        结合BLPD和CCPD数据集的车牌数据集类
        
        Args:
            blpd_txt_path: BLPD数据集的标注文件路径
            ccpd_dirs: CCPD数据集的目录列表 [blur_dir, weather_dir, tilt_dir]
            transform: 图像变换函数
            augment: 是否使用数据增强
            use_yolo: 是否使用YOLO模型检测车牌
            yolo_model: 预训练的YOLO模型
            phase: 训练阶段，'train' 或 'val'
        """
        self.transform = transform
        self.augment = augment
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.phase = phase
        self.samples = []
        
        # 加载BLPD数据集
        if blpd_txt_path and os.path.exists(blpd_txt_path):
            self._load_blpd_dataset(blpd_txt_path)
        
        # 加载CCPD数据集
        if ccpd_dirs:
            for ccpd_dir in ccpd_dirs:
                if os.path.exists(ccpd_dir):
                    self._load_ccpd_dataset(ccpd_dir)
        
        logger.info(f"{phase}数据集共加载 {len(self.samples)} 个有效样本")
        
        # 打印前几个样本示例
        if len(self.samples) > 0:
            logger.info(f"{phase}样本示例:")
            for i in range(min(5, len(self.samples))):
                logger.info(f"  {self.samples[i]}")
    
    def _load_blpd_dataset(self, txt_path):
        """加载BLPD数据集"""
        logger.info(f"加载BLPD数据集: {txt_path}")
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            logger.info(f"读取了 {len(lines)} 行BLPD数据")
            
            valid_count = 0
            invalid_count = 0
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        # 标签是车牌号码字符串（第二列）
                        plate_number = parts[1]
                        
                        # 检查车牌中的字符是否都在我们的字符集中
                        valid = all(char in CHARS for char in plate_number)
                        if valid:
                            self.samples.append(('blpd', os.path.join(BLPD_DIR, image_name), plate_number))
                            valid_count += 1
                        else:
                            # 打印不在字符集中的车牌号
                            invalid_chars = [char for char in plate_number if char not in CHARS]
                            logger.warning(f"跳过含有无效字符的车牌 {plate_number}, 无效字符: {invalid_chars}")
                            invalid_count += 1
            
            logger.info(f"BLPD数据: 有效样本: {valid_count}, 无效样本: {invalid_count}")
                
        except Exception as e:
            logger.error(f"加载BLPD数据集时出错: {e}")
            logger.error(traceback.format_exc())
    
    def _load_ccpd_dataset(self, ccpd_dir):
        """加载CCPD数据集目录"""
        logger.info(f"加载CCPD数据集: {ccpd_dir}")
        
        try:
            image_files = [f for f in os.listdir(ccpd_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"CCPD目录 {os.path.basename(ccpd_dir)} 找到 {len(image_files)} 个图像文件")
            
            valid_count = 0
            invalid_count = 0
            
            for img_file in image_files:
                # 解析文件名获取车牌号码
                plate_number = parse_ccpd_filename(os.path.splitext(img_file)[0])
                
                if plate_number:
                    # 检查车牌中的字符是否都在我们的字符集中
                    valid = all(char in CHARS for char in plate_number)
                    if valid:
                        self.samples.append(('ccpd', os.path.join(ccpd_dir, img_file), plate_number))
                        valid_count += 1
                    else:
                        # 打印不在字符集中的车牌号
                        invalid_chars = [char for char in plate_number if char not in CHARS]
                        logger.warning(f"跳过含有无效字符的车牌 {plate_number}, 无效字符: {invalid_chars}")
                        invalid_count += 1
                else:
                    invalid_count += 1
            
            logger.info(f"CCPD {os.path.basename(ccpd_dir)}: 有效样本: {valid_count}, 无效样本: {invalid_count}")
            
        except Exception as e:
            logger.error(f"加载CCPD数据集 {ccpd_dir} 时出错: {e}")
            logger.error(traceback.format_exc())
    
    def apply_advanced_augmentation(self, image):
        """应用高级数据增强技术"""
        try:
            # 随机选择增强方法
            aug_type = random.randint(0, 5)
            
            if aug_type == 0:
                # 模拟运动模糊
                kernel_size = random.randint(3, 7)
                angle = random.randint(0, 180)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1  # 水平方向的核
                kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(
                    (kernel_size//2, kernel_size//2), angle, 1.0), (kernel_size, kernel_size))
                kernel = kernel / np.sum(kernel)
                image_np = np.array(image)
                blurred = cv2.filter2D(image_np, -1, kernel)
                return Image.fromarray(blurred)
            
            elif aug_type == 1:
                # 随机遮挡
                image_np = np.array(image)
                h, w, _ = image_np.shape
                occlusion_size = (random.randint(5, 15), random.randint(5, 15))
                x = random.randint(0, w - occlusion_size[0])
                y = random.randint(0, h - occlusion_size[1])
                image_np[y:y+occlusion_size[1], x:x+occlusion_size[0], :] = random.randint(0, 255)
                return Image.fromarray(image_np)
            
            elif aug_type == 2:
                # 随机噪声
                image_np = np.array(image).astype(np.float32)
                noise = np.random.normal(0, 15, image_np.shape)
                noisy = np.clip(image_np + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy)
            
            elif aug_type == 3:
                # 随机光照变化
                image_np = np.array(image).astype(np.float32)
                brightness = random.uniform(0.7, 1.3)
                image_np = np.clip(image_np * brightness, 0, 255).astype(np.uint8)
                return Image.fromarray(image_np)
            
            elif aug_type == 4:
                # 模拟雨滴效果
                image_np = np.array(image)
                h, w, _ = image_np.shape
                rain_drops = 20
                for i in range(rain_drops):
                    x = random.randint(0, w-1)
                    y = random.randint(0, h-1)
                    length = random.randint(3, 10)
                    angle = random.randint(70, 110)
                    rr, cc = line(y, x, 
                                min(h-1, int(y + length * math.sin(math.radians(angle)))), 
                                min(w-1, int(x + length * math.cos(math.radians(angle)))))
                    # 确保坐标在图像范围内
                    valid_indices = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    rr, cc = rr[valid_indices], cc[valid_indices]
                    if len(rr) > 0:
                        image_np[rr, cc] = 255  # 白色雨滴
                return Image.fromarray(image_np)
            
            else:
                # 弹性变换（简化版）
                image_np = np.array(image)
                h, w, _ = image_np.shape
                dx = np.random.rand(h, w) * 5 - 2.5
                dy = np.random.rand(h, w) * 5 - 2.5
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                indices_x = np.clip(x + dx, 0, w-1).astype(np.float32)
                indices_y = np.clip(y + dy, 0, h-1).astype(np.float32)
                
                distorted = cv2.remap(image_np, indices_x, indices_y, 
                                    interpolation=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT)
                return Image.fromarray(distorted)
        except Exception as e:
            logger.error(f"数据增强错误: {e}")
            # 出错时返回原始图像
            return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            dataset_type, image_path, plate_number = self.samples[idx]
            
            # 读取图像
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                # 创建一个空白图像作为替代
                dummy_img = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH))
                dummy_label = torch.tensor([0])  # 空标签
                return dummy_img, dummy_img, dummy_label, 0
            
            image = Image.open(image_path).convert('RGB')
            
            # 如果是CCPD数据集且使用YOLO，先检测并裁剪车牌区域
            if dataset_type == 'ccpd' and self.use_yolo and self.yolo_model is not None:
                try:
                    # 使用YOLO检测车牌
                    results = self.yolo_model(image)
                    if len(results[0].boxes) > 0:
                        # 获取置信度最高的框
                        box = results[0].boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # 裁剪图像
                        cropped_img = image.crop((x1, y1, x2, y2))
                        image = cropped_img
                except Exception as e:
                    logger.error(f"使用YOLO检测车牌时出错: {e}")
            
            # 保存原始图像用于去模糊任务
            original_image = image.copy()
            
            # 应用高级数据增强
            if self.augment and random.random() < 0.5:
                image = self.apply_advanced_augmentation(image)
            
            # 应用常规变换
            if self.transform:
                image = self.transform(image)
                original_image = TF.resize(original_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
                original_image = TF.to_tensor(original_image)
                original_image = TF.normalize(
                    original_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            # 转换标签为索引
            label = [CHARS_MAP[c] for c in plate_number]
            label_length = len(label)
            
            return image, original_image, torch.tensor(label), label_length
        except Exception as e:
            logger.error(f"加载样本时出错: {e}, idx={idx}")
            # 创建一个空白图像作为替代
            dummy_img = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH))
            dummy_label = torch.tensor([0])  # 空标签
            return dummy_img, dummy_img, dummy_label, 0

def collate_fn(batch):
    """自定义整理函数，处理不同长度的标签并包括原始图像"""
    try:
        images, original_images, labels, lengths = zip(*batch)
        images = torch.stack(images, 0)
        original_images = torch.stack(original_images, 0)
        
        # 找出最长的标签长度
        max_length = max(lengths)
        
        # 用0填充所有标签到最大长度
        padded_labels = torch.zeros(len(labels), max_length).long()
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        # 转换长度为tensor
        lengths = torch.tensor(lengths)
        
        return images, original_images, padded_labels, lengths
    except Exception as e:
        logger.error(f"整理样本时出错: {e}")
        raise

def get_data_transforms():
    """获取数据转换函数"""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ======== 模型定义 ========
# 1. STN 模块 - 空间变换网络
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        logger.info("初始化STN模块")
        
        # 定位网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算输入到全连接层的特征尺寸
        self.fc_input_size = 128 * (IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8)
        
        # 回归网络
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )
        
        # 初始化权重为单位变换矩阵
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        try:
            batch_size = x.size(0)
            
            # 获取变换参数
            xs = self.localization(x)
            xs = xs.view(batch_size, -1)
            theta = self.fc_loc(xs)
            theta = theta.view(batch_size, 2, 3)
            
            # 应用变换
            grid = nn.functional.affine_grid(theta, x.size(), align_corners=True)
            transformed = nn.functional.grid_sample(x, grid, align_corners=True)
            
            return transformed, theta
        except Exception as e:
            logger.error(f"STN前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# 2. 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        logger.info(f"初始化注意力模块, 输入通道数: {in_channels}")
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        try:
            # 通道注意力
            ca = self.channel_attention(x)
            x = x * ca
            
            # 空间注意力
            sa = self.spatial_attention(x)
            x = x * sa
            
            return x
        except Exception as e:
            logger.error(f"注意力模块前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# 3. 特征提取器选择
def get_feature_extractor(use_mobilenet=USE_MOBILENET, enable_attention=ENABLE_ATTENTION):
    """创建特征提取器"""
    if use_mobilenet:
        return MobileNetFeatureExtractor(pretrained=True, enable_attention=enable_attention)
    else:
        return ResNetFeatureExtractor(pretrained=True, enable_attention=enable_attention)

# 3a. MobileNetV3特征提取器
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, enable_attention=True):
        super(MobileNetFeatureExtractor, self).__init__()
        logger.info(f"初始化MobileNetV3特征提取器, 预训练={pretrained}, 注意力={enable_attention}")
        
        try:
            # 加载预训练的MobileNetV3-Small
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            mobilenet = mobilenet_v3_small(weights=weights)
            
            # 使用前几层作为特征提取器
            self.features = nn.Sequential(
                mobilenet.features[0],  # 第一个卷积层
                mobilenet.features[1],  # 第一个inverted residual block
                mobilenet.features[2],  # 第二个inverted residual block
                mobilenet.features[3],  # 第三个inverted residual block
                mobilenet.features[4],  # 第四个inverted residual block
            )
            
            # 获取通道数
            self.out_channels = 40  # MobileNetV3-Small第4个block输出40个通道
            
            # 添加自定义注意力层
            self.enable_attention = enable_attention
            if enable_attention:
                self.attention = AttentionBlock(self.out_channels)
            
        except Exception as e:
            logger.error(f"初始化MobileNetV3特征提取器时出错: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def forward(self, x):
        try:
            features = self.features(x)
            
            if self.enable_attention:
                features = self.attention(features)
                
            return features
        except Exception as e:
            logger.error(f"MobileNetV3特征提取器前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# 3b. ResNet18特征提取器
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, enable_attention=True):
        super(ResNetFeatureExtractor, self).__init__()
        logger.info(f"初始化ResNet18特征提取器, 预训练={pretrained}, 注意力={enable_attention}")
        
        try:
            # 加载预训练的ResNet18
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            
            # 使用前几层作为特征提取器
            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2
            )
            
            # 获取通道数
            self.out_channels = 128  # ResNet18的layer2输出128个通道
            
            # 添加自定义注意力层
            self.enable_attention = enable_attention
            if enable_attention:
                self.attention = AttentionBlock(self.out_channels)
            
        except Exception as e:
            logger.error(f"初始化ResNet18特征提取器时出错: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def forward(self, x):
        try:
            features = self.features(x)
            
            if self.enable_attention:
                features = self.attention(features)
                
            return features
        except Exception as e:
            logger.error(f"ResNet18特征提取器前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# 4. 去模糊模块
class DeblurModule(nn.Module):
    def __init__(self, in_channels):
        super(DeblurModule, self).__init__()
        logger.info(f"初始化去模糊模块, 输入通道数: {in_channels}")
        
        self.deblur = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出归一化到[0,1]
        )
        
    def forward(self, features, input_size):
        try:
            # 将特征上采样到原始图像尺寸
            deblurred = self.deblur(features)
            deblurred = nn.functional.interpolate(
                deblurred, size=input_size, mode='bilinear', align_corners=True)
            return deblurred
        except Exception as e:
            logger.error(f"去模糊模块前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# 5. 增强的CRNN模型
class EnhancedCRNN(nn.Module):
    def __init__(self, num_classes, enable_stn=ENABLE_STN, enable_deblur=ENABLE_DEBLUR):
        super(EnhancedCRNN, self).__init__()
        logger.info(f"初始化增强CRNN, 类别数={num_classes}, STN={enable_stn}, 去模糊={enable_deblur}")
        
        try:
            # 控制各模块是否启用
            self.enable_stn = enable_stn
            self.enable_deblur = enable_deblur
            
            # STN模块
            self.stn = SpatialTransformer() if enable_stn else None
            
            # 使用特征提取器
            self.cnn = get_feature_extractor(use_mobilenet=USE_MOBILENET, enable_attention=ENABLE_ATTENTION)
            
            # 获取特征的通道数
            self.feature_channels = self.cnn.out_channels
            
            # 去模糊模块
            self.deblur = DeblurModule(self.feature_channels) if enable_deblur else None
            
            # 计算RNN输入尺寸 - 处理MobileNetV3和ResNet18的不同下采样率
            # MobileNetV3下采样率更高，特征图高度更小
            if USE_MOBILENET:
                # MobileNetV3对32x160的输入会产生2x20的特征图
                rnn_input_size = self.feature_channels * 2  # 40 * 2 = 80
            else:
                # ResNet18对32x160的输入会产生4x20的特征图
                rnn_input_size = self.feature_channels * 4  # 128 * 4 = 512
            
            logger.info(f"计算的RNN输入尺寸: {rnn_input_size}")
            
            # 序列建模 (RNN)
            self.rnn_hidden = 256
            self.rnn = nn.LSTM(
                rnn_input_size,  # 动态计算输入特征大小
                self.rnn_hidden, 
                bidirectional=True, 
                batch_first=True
            )
            
            # 分类器
            self.classifier = nn.Linear(self.rnn_hidden * 2, num_classes)
            
            logger.info(f"CRNN初始化完成，特征通道数: {self.feature_channels}")
            
        except Exception as e:
            logger.error(f"初始化CRNN时出错: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def forward(self, x, original_image=None):
        try:
            batch_size = x.size(0)
            
            # 应用STN进行空间变换
            if self.enable_stn:
                x, theta = self.stn(x)
            else:
                theta = None
            
            # 特征提取 (CNN)
            features = self.cnn(x)  # [batch, channels, height, width]
            
            # 去模糊处理 (如果启用)
            if self.enable_deblur and original_image is not None:
                deblurred = self.deblur(features, (IMAGE_HEIGHT, IMAGE_WIDTH))
            else:
                deblurred = None
            
            # 调整尺寸为序列格式
            batch, channels, height, width = features.size()
            logger.debug(f"特征形状: batch={batch}, channels={channels}, height={height}, width={width}")
            
            features = features.permute(0, 3, 1, 2)  # [batch, width, channels, height]
            features = features.reshape(batch, width, channels * height)  # [batch, width, channels*height]
            
            # 序列建模 (RNN)
            rnn_output, _ = self.rnn(features)  # [batch, width, hidden*2]
            
            # 分类
            output = self.classifier(rnn_output)  # [batch, width, num_classes]
            
            return output, deblurred, theta
        except Exception as e:
            logger.error(f"CRNN前向传播出错: {e}")
            logger.error(traceback.format_exc())
            raise

# ======== 训练函数 ========
def train_progressive(use_ccpd=True, use_yolo=True):
    """使用渐进式学习训练端到端车牌识别系统"""
    logger.info("准备训练数据...")
    
    # 获取数据变换
    train_transform, val_transform = get_data_transforms()
    
    # 加载YOLO模型（如果需要）
    yolo_model = None
    if use_yolo and os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"加载YOLO模型: {YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"加载YOLO模型时出错: {e}")
            use_yolo = False  # 如果加载失败，禁用YOLO
    
    # 准备CCPD数据集目录（如果使用）
    ccpd_dirs = []
    if use_ccpd:
        if os.path.exists(CCPD_BLUR_DIR):
            ccpd_dirs.append(CCPD_BLUR_DIR)
        if os.path.exists(CCPD_WEATHER_DIR):
            ccpd_dirs.append(CCPD_WEATHER_DIR)
        if os.path.exists(CCPD_TILT_DIR):
            ccpd_dirs.append(CCPD_TILT_DIR)
    
    # 创建数据集
    train_dataset = CombinedLicensePlateDataset(
        blpd_txt_path=BLPD_TRAIN_TXT,
        ccpd_dirs=ccpd_dirs,
        transform=train_transform,
        augment=True,
        use_yolo=use_yolo,
        yolo_model=yolo_model,
        phase='train'
    )
    
    val_dataset = CombinedLicensePlateDataset(
        blpd_txt_path=BLPD_VAL_TXT,
        ccpd_dirs=None,  # 验证只使用BLPD
        transform=val_transform,
        augment=False,
        use_yolo=use_yolo,
        yolo_model=yolo_model,
        phase='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 保存字符映射，用于后续推理
    chars_mapping = {
        'chars_to_idx': {str(k): v for k, v in CHARS_MAP.items()},
        'idx_to_chars': {str(k): v for k, v in IDX_TO_CHARS.items()},
        'blank_char': BLANK_CHAR,
        'num_classes': NUM_CLASSES
    }
    
    with open(os.path.join(OUTPUT_DIR, 'chars_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(chars_mapping, f, ensure_ascii=False, indent=4)
    logger.info("字符映射已保存到 chars_mapping.json")
    
    # 检查是否需要恢复训练
    start_stage = args.stage
    resume_epoch = 0
    best_models = {}
    
    if args.resume:
        config = load_resume_config()
        if config:
            start_stage = config['stage']
            resume_epoch = config['epoch'] + 1  # 从下一个epoch开始
            
            # 加载之前阶段的最佳模型
            for stage in range(1, start_stage):
                stage_best_model_path = os.path.join(OUTPUT_DIR, f'{STAGE_NAMES[stage-1]}_best_model.pth')
                if os.path.exists(stage_best_model_path):
                    best_models[stage] = torch.load(stage_best_model_path, map_location=device)
                    logger.info(f"已加载第{stage}阶段最佳模型: {stage_best_model_path}")
    
    # ======== 渐进式学习阶段 ========
    # 全局开始时间
    global global_start_time
    global_start_time = time.time()
    
    # 阶段1: 基础CRNN训练（没有STN和去模糊）
    if start_stage <= 1:
        logger.info("=== 阶段1: 基础CRNN训练 ===")
        model_stage1 = EnhancedCRNN(
            num_classes=NUM_CLASSES,
            enable_stn=False,
            enable_deblur=False
        ).to(device)
        
        # 如果从某个检查点恢复
        checkpoint_to_load = None
        if args.resume and start_stage == 1 and resume_epoch > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'stage1_checkpoint_epoch_{resume_epoch-1}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint_to_load = torch.load(checkpoint_path, map_location=device)
                logger.info(f"从检查点恢复阶段1: {checkpoint_path}, 开始于 epoch {resume_epoch}")
        
        best_model_stage1 = train_one_stage(
            model_stage1, train_loader, val_loader, STAGE1_EPOCHS, 
            "stage1", enable_stn=False, enable_deblur=False,
            start_epoch=resume_epoch, checkpoint=checkpoint_to_load
        )
        
        best_models[1] = best_model_stage1
        start_stage = 2
        resume_epoch = 0
    
    # 阶段2: 添加STN
    if start_stage <= 2:
        logger.info("=== 阶段2: 添加STN ===")
        model_stage2 = EnhancedCRNN(
            num_classes=NUM_CLASSES,
            enable_stn=True,
            enable_deblur=False
        ).to(device)
        
        # 从阶段1加载权重，除了STN部分
        if 1 in best_models:
            pretrained_dict = best_models[1]['model_state_dict']
            model_dict = model_stage2.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('stn')}
            model_dict.update(pretrained_dict)
            model_stage2.load_state_dict(model_dict)
            logger.info("从阶段1加载模型权重")
        
        # 如果从某个检查点恢复
        checkpoint_to_load = None
        if args.resume and start_stage == 2 and resume_epoch > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'stage2_checkpoint_epoch_{resume_epoch-1}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint_to_load = torch.load(checkpoint_path, map_location=device)
                logger.info(f"从检查点恢复阶段2: {checkpoint_path}, 开始于 epoch {resume_epoch}")
        
        best_model_stage2 = train_one_stage(
            model_stage2, train_loader, val_loader, STAGE2_EPOCHS, 
            "stage2", enable_stn=True, enable_deblur=False,
            start_epoch=resume_epoch, checkpoint=checkpoint_to_load
        )
        
        best_models[2] = best_model_stage2
        start_stage = 3
        resume_epoch = 0
    
    # 阶段3: 添加去模糊
    if start_stage <= 3:
        logger.info("=== 阶段3: 添加去模糊 ===")
        model_stage3 = EnhancedCRNN(
            num_classes=NUM_CLASSES,
            enable_stn=True,
            enable_deblur=True
        ).to(device)
        
        # 从阶段2加载权重，除了去模糊部分
        if 2 in best_models:
            pretrained_dict = best_models[2]['model_state_dict']
            model_dict = model_stage3.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('deblur')}
            model_dict.update(pretrained_dict)
            model_stage3.load_state_dict(model_dict)
            logger.info("从阶段2加载模型权重")
        
        # 如果从某个检查点恢复
        checkpoint_to_load = None
        if args.resume and start_stage == 3 and resume_epoch > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'stage3_checkpoint_epoch_{resume_epoch-1}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint_to_load = torch.load(checkpoint_path, map_location=device)
                logger.info(f"从检查点恢复阶段3: {checkpoint_path}, 开始于 epoch {resume_epoch}")
        
        best_model_stage3 = train_one_stage(
            model_stage3, train_loader, val_loader, STAGE3_EPOCHS, 
            "stage3", enable_stn=True, enable_deblur=True, 
            deblur_weight=DEBLUR_WEIGHT_INITIAL,
            start_epoch=resume_epoch, checkpoint=checkpoint_to_load
        )
        
        best_models[3] = best_model_stage3
        start_stage = 4
        resume_epoch = 0
    
    # 阶段4: 联合微调
    if start_stage <= 4:
        logger.info("=== 阶段4: 联合微调 ===")
        model_stage4 = EnhancedCRNN(
            num_classes=NUM_CLASSES,
            enable_stn=True,
            enable_deblur=True
        ).to(device)
        
        # 从阶段3加载权重
        if 3 in best_models:
            model_stage4.load_state_dict(best_models[3]['model_state_dict'])
            logger.info("从阶段3加载模型权重")
        
        # 如果从某个检查点恢复
        checkpoint_to_load = None
        if args.resume and start_stage == 4 and resume_epoch > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'stage4_checkpoint_epoch_{resume_epoch-1}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint_to_load = torch.load(checkpoint_path, map_location=device)
                logger.info(f"从检查点恢复阶段4: {checkpoint_path}, 开始于 epoch {resume_epoch}")
        
        best_model_stage4 = train_one_stage(
            model_stage4, train_loader, val_loader, STAGE4_EPOCHS,
            "stage4", enable_stn=True, enable_deblur=True,
            deblur_weight=DEBLUR_WEIGHT_FINAL,
            start_epoch=resume_epoch, checkpoint=checkpoint_to_load
        )
        
        best_models[4] = best_model_stage4
    
    # 保存最终模型
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    if 4 in best_models:
        torch.save(best_models[4], final_model_path)
        logger.info(f"保存最终模型到 {final_model_path}")
    elif 3 in best_models:
        torch.save(best_models[3], final_model_path)
        logger.info(f"保存最终模型(阶段3)到 {final_model_path}")
    
    # 清除暂停文件
    clean_pause_file()
    
    return best_models.get(4, best_models.get(3))

def train_one_stage(model, train_loader, val_loader, num_epochs, stage_name, 
                    enable_stn=True, enable_deblur=True, deblur_weight=0.3,
                    start_epoch=0, checkpoint=None):
    """训练模型的一个阶段"""
    logger.info(f"开始训练 {stage_name}，epochs={num_epochs}，开始于epoch={start_epoch}")
    
    # 定义损失函数和优化器
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    mse_loss = nn.MSELoss()
    
    # 如果有检查点，恢复优化器状态
    if checkpoint is not None:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"从检查点恢复优化器状态")
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"从检查点恢复模型权重")
        
        # 加载最佳验证损失
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"从检查点恢复的最佳验证损失: {best_val_loss:.4f}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_val_loss = float('inf')
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练循环
    best_model_info = None
    early_stopping_patience = 10  # 早停耐心值
    early_stopping_counter = 0
    early_stopping_flag = False
    stage_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        # 如果早停标志被触发，则退出训练循环
        if early_stopping_flag:
            logger.info(f"{stage_name} 应用早停策略，提前结束训练")
            break
        
        # 检查是否应该暂停训练
        if should_pause_training():
            # 保存当前训练状态
            current_checkpoint_path = os.path.join(OUTPUT_DIR, f'{stage_name}_checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'stage': stage_name
            }, current_checkpoint_path)
            
            # 保存恢复配置
            save_resume_config(
                list(STAGE_NAMES).index(stage_name) + 1, 
                epoch, 
                best_model_info
            )
            
            logger.info(f"训练暂停在 {stage_name} 阶段，epoch {epoch}")
            logger.info(f"保存检查点到 {current_checkpoint_path}")
            logger.info(f"要恢复训练，请使用 --resume 选项运行脚本")
            return best_model_info if best_model_info else {'model_state_dict': model.state_dict()}
            
        logger.info(f"{stage_name} 开始第 {epoch+1}/{num_epochs} 轮训练")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_recognition_loss = 0.0
        train_deblur_loss = 0.0
        
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (images, original_images, targets, target_lengths) in enumerate(progress_bar):
            images = images.to(device)
            original_images = original_images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits, deblurred, _ = model(images, original_images)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            
            # 计算输入长度（假设所有序列具有相同长度）
            batch_size, width, _ = logits.size()
            input_lengths = torch.full((batch_size,), width, dtype=torch.long, device=device)
            
            # 计算识别（CTC）损失
            recognition_loss = ctc_loss(
                log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            
            # 计算去模糊损失（如果启用）
            if enable_deblur and deblurred is not None:
                deblur_loss = mse_loss(deblurred, original_images)
                loss = recognition_loss + deblur_weight * deblur_loss
            else:
                deblur_loss = torch.tensor(0.0).to(device)
                loss = recognition_loss
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # 更新损失统计
            train_loss += loss.item()
            train_recognition_loss += recognition_loss.item()
            train_deblur_loss += deblur_loss.item() if enable_deblur and deblurred is not None else 0
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'rec_loss': f"{recognition_loss.item():.4f}",
                'deblur_loss': f"{deblur_loss.item():.4f}" if enable_deblur and deblurred is not None else "N/A"
            })
            
            # 每100个批次记录训练指标
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar(f'{stage_name}/Train/Loss', loss.item(), step)
                writer.add_scalar(f'{stage_name}/Train/RecognitionLoss', recognition_loss.item(), step)
                if enable_deblur and deblurred is not None:
                    writer.add_scalar(f'{stage_name}/Train/DeblurLoss', deblur_loss.item(), step)
            
            # 检查是否应该暂停训练
            if batch_idx % 10 == 0 and should_pause_training():
                # 创建临时检查点路径
                temp_checkpoint_path = os.path.join(OUTPUT_DIR, f'{stage_name}_temp_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, temp_checkpoint_path)
                
                # 保存恢复配置
                save_resume_config(
                    list(STAGE_NAMES).index(stage_name) + 1, 
                    epoch, 
                    best_model_info
                )
                
                logger.info(f"训练暂停在 {stage_name} 阶段，epoch {epoch}，batch {batch_idx}")
                logger.info(f"保存临时检查点到 {temp_checkpoint_path}")
                logger.info(f"要恢复训练，请使用 --resume 选项运行脚本")
                return best_model_info if best_model_info else {'model_state_dict': model.state_dict()}
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_rec_loss = train_recognition_loss / len(train_loader)
        avg_train_deblur_loss = train_deblur_loss / len(train_loader) if enable_deblur else 0
        
        train_time = time.time() - epoch_start_time
        logger.info(f"{stage_name} 训练耗时: {train_time:.2f}秒")
        logger.info(f"{stage_name} 平均训练损失: {avg_train_loss:.4f}, 识别损失: {avg_train_rec_loss:.4f}, "
                   f"去模糊损失: {avg_train_deblur_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_recognition_loss = 0.0
        val_deblur_loss = 0.0
        correct_chars = 0
        total_chars = 0
        correct_plates = 0
        total_plates = 0
        
        val_start_time = time.time()
        progress_bar = tqdm(val_loader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, original_images, targets, target_lengths) in enumerate(progress_bar):
                images = images.to(device)
                original_images = original_images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                # 前向传播
                logits, deblurred, _ = model(images, original_images)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                
                # 计算输入长度
                batch_size, width, _ = logits.size()
                input_lengths = torch.full((batch_size,), width, dtype=torch.long, device=device)
                
                # 计算识别（CTC）损失
                recognition_loss = ctc_loss(
                    log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                
                # 计算去模糊损失（如果启用）
                if enable_deblur and deblurred is not None:
                    deblur_loss = mse_loss(deblurred, original_images)
                    loss = recognition_loss + deblur_weight * deblur_loss
                else:
                    deblur_loss = torch.tensor(0.0).to(device)
                    loss = recognition_loss
                
                # 更新损失统计
                val_loss += loss.item()
                val_recognition_loss += recognition_loss.item()
                val_deblur_loss += deblur_loss.item() if enable_deblur and deblurred is not None else 0
                
                # 解码预测结果并计算准确率
                _, predictions = log_probs.max(2)
                predictions = predictions.cpu().numpy()
                
                for i, (pred, target_length) in enumerate(zip(predictions, target_lengths)):
                    decoded_pred = []
                    prev = -1
                    
                    # CTC解码（简单贪婪解码）
                    for p in pred:
                        if p != 0 and p != prev:  # 不是blank且不重复
                            decoded_pred.append(p)
                        prev = p
                    
                    # 获取真实标签
                    true_label = targets[i][:target_length].cpu().numpy()
                    
                    # 转换为文本进行比较
                    decoded_text = ''.join([IDX_TO_CHARS.get(idx, BLANK_CHAR) for idx in decoded_pred])
                    true_text = ''.join([IDX_TO_CHARS.get(idx, BLANK_CHAR) for idx in true_label])
                    
                    # 计算字符级准确率
                    min_len = min(len(decoded_pred), len(true_label))
                    correct_chars += sum(1 for j in range(min_len) if decoded_pred[j] == true_label[j])
                    total_chars += len(true_label)
                    
                    # 计算完整车牌匹配率
                    if decoded_text == true_text:
                        correct_plates += 1
                    total_plates += 1
                    
                    # 记录一些样本的预测结果
                    if batch_idx == 0 and i < 5:  # 只记录第一个批次的前5个样本
                        logger.info(f"{stage_name} Sample {i}: Pred '{decoded_text}', True '{true_text}'")
                
                # 更新进度条
                char_acc = correct_chars / total_chars if total_chars > 0 else 0
                plate_acc = correct_plates / total_plates if total_plates > 0 else 0
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'char_acc': f"{char_acc:.4f}",
                    'plate_acc': f"{plate_acc:.4f}"
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rec_loss = val_recognition_loss / len(val_loader)
        avg_val_deblur_loss = val_deblur_loss / len(val_loader) if enable_deblur else 0
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        plate_accuracy = correct_plates / total_plates if total_plates > 0 else 0
        
        val_time = time.time() - val_start_time
        logger.info(f"{stage_name} 验证耗时: {val_time:.2f}秒")
        logger.info(f"{stage_name} 平均验证损失: {avg_val_loss:.4f}, 识别损失: {avg_val_rec_loss:.4f}, "
                   f"去模糊损失: {avg_val_deblur_loss:.4f}")
        logger.info(f"{stage_name} 字符准确率: {char_accuracy:.4f} ({correct_chars}/{total_chars})")
        logger.info(f"{stage_name} 车牌准确率: {plate_accuracy:.4f} ({correct_plates}/{total_plates})")
        
        # 写入TensorBoard
        writer.add_scalar(f'{stage_name}/Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar(f'{stage_name}/Validation/RecognitionLoss', avg_val_rec_loss, epoch)
        if enable_deblur:
            writer.add_scalar(f'{stage_name}/Validation/DeblurLoss', avg_val_deblur_loss, epoch)
        writer.add_scalar(f'{stage_name}/Validation/CharAccuracy', char_accuracy, epoch)
        writer.add_scalar(f'{stage_name}/Validation/PlateAccuracy', plate_accuracy, epoch)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # 重置早停计数器
            model_path = os.path.join(OUTPUT_DIR, f'{stage_name}_best_model.pth')
            
            # 创建模型信息字典
            best_model_info = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'char_accuracy': char_accuracy,
                'plate_accuracy': plate_accuracy,
                'stage': stage_name
            }
            
            torch.save(best_model_info, model_path)
            logger.info(f"保存 {stage_name} 阶段最佳模型到 {model_path}")
        else:
            early_stopping_counter += 1
            logger.info(f"{stage_name} 验证损失未改善，早停计数: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"{stage_name} 早停触发！连续{early_stopping_patience}个epoch验证损失未改善")
                early_stopping_flag = True
        
        # 保存阶段检查点
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'{stage_name}_checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'char_accuracy': char_accuracy,
                'plate_accuracy': plate_accuracy,
                'stage': stage_name
            }, checkpoint_path)
            logger.info(f"保存 {stage_name} 检查点到 {checkpoint_path}")
    
    stage_time = time.time() - stage_start_time
    logger.info(f"{stage_name} 阶段训练完成! 总耗时: {stage_time/60:.2f} 分钟")
    logger.info(f"{stage_name} 最佳验证损失: {best_val_loss:.4f}")
    
    return best_model_info

# ======== 评估函数 ========
def evaluate_model(model, test_loader):
    """评估模型性能"""
    logger.info("开始模型评估...")
    
    # 设置为评估模式
    model.eval()
    
    # 统计指标
    correct_plates = 0
    total_plates = 0
    correct_chars = 0
    total_chars = 0
    
    # 用于可视化的结果收集
    visualization_samples = []
    deblur_samples = []
    
    # 评估循环
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, (images, original_images, targets, target_lengths) in enumerate(progress_bar):
            images = images.to(device)
            original_images = original_images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            logits, deblurred, theta = model(images, original_images)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            
            # 解码预测结果
            _, predictions = log_probs.max(2)
            predictions = predictions.cpu().numpy()
            
            for i, (pred, target_length) in enumerate(zip(predictions, target_lengths)):
                # 解码预测结果（贪婪解码）
                decoded_pred = []
                prev = -1
                for p in pred:
                    if p != 0 and p != prev:  # 不是blank且不重复
                        decoded_pred.append(p)
                    prev = p
                
                # 获取真实标签
                true_label = targets[i][:target_length].cpu().numpy()
                
                # 转换为文本
                pred_text = ''.join([IDX_TO_CHARS.get(idx, BLANK_CHAR) for idx in decoded_pred])
                true_text = ''.join([IDX_TO_CHARS.get(idx, BLANK_CHAR) for idx in true_label])
                
                # 统计字符级准确率
                min_len = min(len(decoded_pred), len(true_label))
                chars_correct_in_sample = sum(1 for j in range(min_len) if decoded_pred[j] == true_label[j])
                correct_chars += chars_correct_in_sample
                total_chars += len(true_label)
                
                # 统计完整车牌匹配率
                is_correct = (pred_text == true_text)
                if is_correct:
                    correct_plates += 1
                total_plates += 1
                
                # 收集可视化样本
                if len(visualization_samples) < 20 and (batch_idx % 5 == 0 or is_correct == False):
                    # 转换图像为可显示格式
                    img_np = images[i].cpu().detach().numpy().transpose(1, 2, 0)
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    visualization_samples.append({
                        'image': img_np,
                        'pred': pred_text,
                        'true': true_text,
                        'is_correct': is_correct,
                        'char_accuracy': chars_correct_in_sample / len(true_label) if len(true_label) > 0 else 0
                    })
                
                # 收集去模糊结果
                if deblurred is not None and len(deblur_samples) < 10 and batch_idx % 10 == 0:
                    original_np = original_images[i].cpu().detach().numpy().transpose(1, 2, 0)
                    original_np = (original_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    original_np = np.clip(original_np, 0, 255).astype(np.uint8)
                    
                    deblur_np = deblurred[i].cpu().detach().numpy().transpose(1, 2, 0) * 255
                    deblur_np = np.clip(deblur_np, 0, 255).astype(np.uint8)
                    
                    deblur_samples.append({
                        'original': original_np,
                        'deblurred': deblur_np,
                        'pred': pred_text,
                        'true': true_text
                    })
            
            # 更新进度条
            plate_accuracy = correct_plates / total_plates if total_plates > 0 else 0
            char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
            progress_bar.set_postfix({
                'plate_acc': f"{plate_accuracy:.4f}",
                'char_acc': f"{char_accuracy:.4f}"
            })
    
    # 计算总体准确率
    plate_accuracy = correct_plates / total_plates if total_plates > 0 else 0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    logger.info(f"评估完成!")
    logger.info(f"车牌准确率: {plate_accuracy:.4f} ({correct_plates}/{total_plates})")
    logger.info(f"字符准确率: {char_accuracy:.4f} ({correct_chars}/{total_chars})")
    
    # 保存评估结果
    results = {
        'plate_accuracy': plate_accuracy,
        'char_accuracy': char_accuracy,
        'correct_plates': correct_plates,
        'total_plates': total_plates,
        'correct_chars': correct_chars,
        'total_chars': total_chars
    }
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 可视化预测结果
    visualize_predictions(visualization_samples)
    
    # 可视化去模糊结果
    if len(deblur_samples) > 0:
        visualize_deblur_results(deblur_samples)
    
    return results

def visualize_predictions(samples):
    """可视化预测结果"""
    if not samples:
        logger.warning("没有样本可供可视化")
        return
    
    num_samples = len(samples)
    rows = (num_samples + 3) // 4  # 每行最多4个样本
    
    plt.figure(figsize=(16, 4 * rows))
    
    for i, sample in enumerate(samples):
        plt.subplot(rows, 4, i+1)
        plt.imshow(sample['image'])
        plt.title(f"Pred: {sample['pred']}\nTrue: {sample['true']}", 
                 color='green' if sample['is_correct'] else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_results.png'), dpi=200)
    logger.info(f"可视化结果已保存到 {os.path.join(OUTPUT_DIR, 'prediction_results.png')}")

def visualize_deblur_results(samples):
    """可视化去模糊结果"""
    if not samples:
        logger.warning("没有去模糊样本可供可视化")
        return
    
    num_samples = len(samples)
    plt.figure(figsize=(12, 4 * num_samples))
    
    for i, sample in enumerate(samples):
        # 原始图像
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(sample['original'])
        plt.title(f"Original - {sample['true']}")
        plt.axis('off')
        
        # 去模糊后图像
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(sample['deblurred'])
        plt.title(f"Deblurred - {sample['pred']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deblur_results.png'), dpi=200)
    logger.info(f"去模糊可视化结果已保存到 {os.path.join(OUTPUT_DIR, 'deblur_results.png')}")

# ======== 创建暂停文件函数 ========
def create_pause_file():
    """创建暂停文件"""
    try:
        with open(PAUSE_FILE, 'w') as f:
            f.write(f"训练暂停于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"创建暂停文件: {PAUSE_FILE}")
        return True
    except Exception as e:
        logger.error(f"创建暂停文件时出错: {e}")
        return False

# ======== 主函数 ========
def main():
    # 在Windows上运行多进程程序需要这个保护
    if sys.platform.startswith('win'):
        mp.freeze_support()
    
    # 清除可能存在的暂停文件
    if not args.resume:
        clean_pause_file()
    
    logger.info("开始端到端车牌识别系统训练")
    
    # 检查预训练的YOLO模型
    if os.path.exists(YOLO_MODEL_PATH):
        logger.info(f"发现预训练的YOLO模型: {YOLO_MODEL_PATH}")
    else:
        logger.warning(f"未找到预训练的YOLO模型: {YOLO_MODEL_PATH}")
    
    # 渐进式训练
    best_model = train_progressive(use_ccpd=True, use_yolo=True)
    
    # 如果训练被暂停，则不进行评估
    if should_pause_training():
        logger.info("训练已暂停，跳过评估")
        return
    
    # 创建验证数据集和加载器用于最终评估
    _, val_transform = get_data_transforms()
    
    val_dataset = CombinedLicensePlateDataset(
        blpd_txt_path=BLPD_VAL_TXT,
        ccpd_dirs=None,  # 只使用BLPD
        transform=val_transform,
        augment=False,
        use_yolo=False,  # 评估时不使用YOLO
        yolo_model=None,
        phase='test'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 加载最佳模型并评估
    model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    if os.path.exists(model_path):
        model_info = torch.load(model_path)
        
        model = EnhancedCRNN(
            num_classes=NUM_CLASSES,
            enable_stn=True,
            enable_deblur=True
        ).to(device)
        
        model.load_state_dict(model_info['model_state_dict'])
        logger.info(f"加载最终模型: {model_path}")
        
        # 最终评估
        evaluate_model(model, val_loader)
    else:
        logger.error(f"未找到最终模型: {model_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("检测到手动中断，创建暂停文件并保存当前状态")
        create_pause_file()
        print("\n训练已暂停。要继续训练，请使用 --resume 选项运行脚本")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.error(traceback.format_exc())
        # 发生错误时尝试创建暂停文件
        create_pause_file()