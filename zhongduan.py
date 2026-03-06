#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端车牌识别系统
基于YOLO+STN+CRNN+去模糊的端到端车牌识别系统
支持UI界面可视化
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import traceback
import json
from torch.nn import functional as F
import sys
import io

class FakeStdout(io.StringIO):
    @property
    def encoding(self):
        return 'utf-8'

# 伪造 stdout 和 stderr，防止 ultralytics 报错
if not sys.stdout:
    sys.stdout = FakeStdout()
if not sys.stderr:
    sys.stderr = FakeStdout()



# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型路径和配置
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
LPR_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pth")
CHARS_MAPPING_PATH = os.path.join(MODEL_DIR, "chars_mapping.json")

# 图像预处理参数
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160

# 加载字符映射
def load_chars_mapping():
    """加载字符映射文件"""
    if not os.path.exists(CHARS_MAPPING_PATH):
        # 如果没有映射文件，创建默认的中文字符集
        PROVINCES = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽", "赣", 
                 "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", 
                 "青", "宁", "新", "港", "澳", "台"]
        ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        DIGITS = [str(i) for i in range(10)]
        CHARS = PROVINCES + ALPHABETS + DIGITS
        
        BLANK_CHAR = '-'
        CHARS_MAP = {char: i+1 for i, char in enumerate(CHARS)}
        IDX_TO_CHARS = {i+1: char for i, char in enumerate(CHARS)}
        IDX_TO_CHARS[0] = BLANK_CHAR
        NUM_CLASSES = len(CHARS) + 1
        
        chars_mapping = {
            'chars_to_idx': {str(k): v for k, v in CHARS_MAP.items()},
            'idx_to_chars': {str(k): v for k, v in IDX_TO_CHARS.items()},
            'blank_char': BLANK_CHAR,
            'num_classes': NUM_CLASSES
        }
        
        # 保存映射
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CHARS_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(chars_mapping, f, ensure_ascii=False, indent=4)
        
        return chars_mapping
    
    # 加载已有的映射文件
    with open(CHARS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        chars_mapping = json.load(f)
    
    # 将字符串键转为整数键
    idx_to_chars = {int(k): v for k, v in chars_mapping['idx_to_chars'].items()}
    chars_mapping['idx_to_chars'] = idx_to_chars
    
    return chars_mapping

# 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        
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
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

# MobileNetV3特征提取器
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, enable_attention=True):
        super(MobileNetFeatureExtractor, self).__init__()
        
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
            
    def forward(self, x):
        features = self.features(x)
        
        if self.enable_attention:
            features = self.attention(features)
            
        return features

# STN 模块 - 空间变换网络
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        
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
        batch_size = x.size(0)
        
        # 获取变换参数
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(batch_size, 2, 3)
        
        # 应用变换
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        transformed = F.grid_sample(x, grid, align_corners=True)
        
        return transformed, theta

# 去模糊模块
class DeblurModule(nn.Module):
    def __init__(self, in_channels):
        super(DeblurModule, self).__init__()
        
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
        # 将特征上采样到原始图像尺寸
        deblurred = self.deblur(features)
        deblurred = F.interpolate(
            deblurred, size=input_size, mode='bilinear', align_corners=True)
        return deblurred

# 增强的CRNN模型
class EnhancedCRNN(nn.Module):
    def __init__(self, num_classes, enable_stn=True, enable_deblur=True):
        super(EnhancedCRNN, self).__init__()
        
        # 控制各模块是否启用
        self.enable_stn = enable_stn
        self.enable_deblur = enable_deblur
        
        # STN模块
        self.stn = SpatialTransformer() if enable_stn else None
        
        # 使用特征提取器
        self.cnn = MobileNetFeatureExtractor(pretrained=True, enable_attention=True)
        
        # 获取特征的通道数
        self.feature_channels = self.cnn.out_channels
        
        # 去模糊模块
        self.deblur = DeblurModule(self.feature_channels) if enable_deblur else None
        
        # MobileNetV3对32x160的输入会产生2x20的特征图
        rnn_input_size = self.feature_channels * 2  # 40 * 2 = 80
        
        # 序列建模 (RNN)
        self.rnn_hidden = 256
        self.rnn = nn.LSTM(
            rnn_input_size,
            self.rnn_hidden, 
            bidirectional=True, 
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Linear(self.rnn_hidden * 2, num_classes)
        
    def forward(self, x, original_image=None):
        # 保存输入图像用于可视化
        input_image = x.clone()
        
        batch_size = x.size(0)
        
        # 应用STN进行空间变换
        if self.enable_stn:
            x, theta = self.stn(x)
            stn_output = x.clone()
        else:
            theta = None
            stn_output = None
        
        # 特征提取 (CNN)
        features = self.cnn(x)  # [batch, channels, height, width]
        feature_maps = features.clone()
        
        # 去模糊处理 (如果启用)
        if self.enable_deblur and original_image is not None:
            deblurred = self.deblur(features, (IMAGE_HEIGHT, IMAGE_WIDTH))
        else:
            deblurred = None
        
        # 调整尺寸为序列格式
        batch, channels, height, width = features.size()
        
        features = features.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        features = features.reshape(batch, width, channels * height)  # [batch, width, channels*height]
        
        # 序列建模 (RNN)
        rnn_output, _ = self.rnn(features)  # [batch, width, hidden*2]
        
        # 分类
        output = self.classifier(rnn_output)  # [batch, width, num_classes]
        
        # 返回所有中间结果用于可视化
        return output, deblurred, theta, input_image, stn_output, feature_maps

class LicensePlateRecognitionSystem:
    """车牌识别系统主类，集成YOLO检测和CRNN识别"""
    def __init__(self, root):
        self.root = root
        self.root.title("端到端车牌识别系统")
        self.root.geometry("1200x800")
        
        # 加载字符映射
        self.chars_mapping = load_chars_mapping()
        self.idx_to_chars = self.chars_mapping['idx_to_chars']
        self.num_classes = self.chars_mapping.get('num_classes', 75)  # 默认75类
        
        # 初始化变量
        self.input_image = None
        self.lp_image = None
        self.input_image_cv = None
        self.plate_image_tensor = None
        self.plate_boxes = None
        
        # 创建布局
        self.create_widgets()
        
        # 加载模型
        self.load_models()
        
    def create_widgets(self):
        # 创建按钮框架
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 添加按钮
        self.load_btn = tk.Button(btn_frame, text="加载图像", command=self.load_image, width=15, height=2)
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        self.recognize_btn = tk.Button(btn_frame, text="识别车牌", command=self.recognize_plate, width=15, height=2)
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        self.recognize_btn.config(state=tk.DISABLED)  # 初始禁用
        
        # 图像显示区域
        img_frame = tk.Frame(self.root)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧原始图像
        left_frame = tk.Frame(img_frame, width=580, height=600)
        left_frame.pack(side=tk.LEFT, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        left_label = tk.Label(left_frame, text="输入图像")
        left_label.pack(pady=5)
        
        self.input_image_canvas = tk.Canvas(left_frame, bg="lightgray", width=560, height=560)
        self.input_image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧车牌图像
        right_frame = tk.Frame(img_frame, width=580, height=600)
        right_frame.pack(side=tk.LEFT, padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        right_label = tk.Label(right_frame, text="车牌区域")
        right_label.pack(pady=5)
        
        self.plate_image_canvas = tk.Canvas(right_frame, bg="lightgray", width=560, height=420)
        self.plate_image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 识别结果显示
        result_frame = tk.Frame(right_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        result_label = tk.Label(result_frame, text="识别结果:")
        result_label.pack(side=tk.LEFT, padx=5)
        
        self.result_var = tk.StringVar()
        self.result_var.set("未识别")
        result_display = tk.Label(result_frame, textvariable=self.result_var, 
                                  font=("Arial", 24, "bold"), fg="blue")
        result_display.pack(side=tk.LEFT, padx=10)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_models(self):
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            
            # 加载YOLO模型
            if os.path.exists(YOLO_MODEL_PATH):
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                print(f"YOLO模型已加载: {YOLO_MODEL_PATH}")
            else:
                # 如果没有找到本地模型，使用YOLO的最新版本
                self.yolo_model = YOLO("yolov8n.pt")
                print("使用默认YOLO模型")
            
            # 加载车牌识别模型
            if os.path.exists(LPR_MODEL_PATH):
                model_info = torch.load(LPR_MODEL_PATH, map_location=device)
                
                # 创建模型实例
                self.lpr_model = EnhancedCRNN(
                    num_classes=self.num_classes,
                    enable_stn=True,
                    enable_deblur=True
                ).to(device)
                
                # 加载权重
                if 'model_state_dict' in model_info:
                    self.lpr_model.load_state_dict(model_info['model_state_dict'])
                else:
                    self.lpr_model.load_state_dict(model_info)
                
                self.lpr_model.eval()
                print(f"车牌识别模型已加载: {LPR_MODEL_PATH}")
            else:
                messagebox.showwarning("模型缺失", 
                                      f"找不到车牌识别模型: {LPR_MODEL_PATH}\n只能进行车牌检测，无法识别")
                self.lpr_model = None
            
            # 预处理转换
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.status_var.set("模型加载完成")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            traceback.print_exc()
            self.status_var.set("模型加载失败")
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 重置状态
            self.plate_image_tensor = None
            self.plate_boxes = None
            self.result_var.set("未识别")
            
            # 加载并显示图像
            self.input_image_cv = cv2.imread(file_path)
            if self.input_image_cv is None:
                raise ValueError(f"无法读取图像: {file_path}")
            
            # BGR转RGB
            image_rgb = cv2.cvtColor(self.input_image_cv, cv2.COLOR_BGR2RGB)
            
            # 调整图像大小以适应显示区域，保持纵横比
            height, width = image_rgb.shape[:2]
            max_size = 560
            
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # 使用PIL进行缩放
            self.input_image = Image.fromarray(image_rgb)
            display_image = self.input_image.resize((new_width, new_height), Image.LANCZOS)
            self.input_tk = ImageTk.PhotoImage(display_image)
            
            # 在画布上显示
            self.input_image_canvas.config(width=new_width, height=new_height)
            self.input_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.input_tk)
            
            # 清除车牌区域显示
            self.plate_image_canvas.delete("all")
            
            # 启用识别按钮
            self.recognize_btn.config(state=tk.NORMAL)
            
            # 更新状态
            self.status_var.set(f"已加载图像: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"加载图像时出错: {e}")
            traceback.print_exc()
            messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
    
    def recognize_plate(self):
        if self.input_image_cv is None:
            messagebox.showinfo("提示", "请先加载图像")
            return
        
        try:
            self.status_var.set("正在进行车牌检测与识别...")
            self.root.update()
            
            # 使用YOLO检测车牌
            results = self.yolo_model(self.input_image_cv)
            
            # 处理检测结果
            if len(results[0].boxes) == 0:
                self.status_var.set("未检测到车牌")
                messagebox.showinfo("结果", "未检测到车牌")
                return
            
            # 获取所有车牌框
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
            # 保存所有车牌框
            self.plate_boxes = []
            
            # 找出得分最高的车牌
            best_box_idx = np.argmax(scores)
            best_box = boxes[best_box_idx].astype(int)
            
            # 裁剪车牌区域
            x1, y1, x2, y2 = best_box
            plate_img = self.input_image_cv[y1:y2, x1:x2]
            
            # 保存车牌框信息，在原始图像上绘制标注并显示
            self.plate_boxes.append(best_box)
            self._draw_detection_results()
            
            # BGR转RGB用于显示
            plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            self.lp_image = Image.fromarray(plate_img_rgb)
            
            # 调整大小以适应显示区域，保持纵横比
            height, width = plate_img_rgb.shape[:2]
            max_width, max_height = 560, 420
            
            if width / height > max_width / max_height:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))
            
            # 显示车牌图像
            display_plate = self.lp_image.resize((new_width, new_height), Image.LANCZOS)
            self.plate_tk = ImageTk.PhotoImage(display_plate)
            
            self.plate_image_canvas.config(width=new_width, height=new_height)
            self.plate_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.plate_tk)
            
            # 预处理车牌图像，准备识别
            if self.lpr_model is not None:
                # 转换为模型输入格式
                plate_tensor = self.transform(self.lp_image).unsqueeze(0).to(device)
                self.plate_image_tensor = plate_tensor
                
                # 识别车牌文本
                plate_number = self._recognize_plate_text(plate_tensor)
                
                # 显示识别结果
                self.result_var.set(plate_number)
                self.status_var.set(f"识别完成: {plate_number}")
            else:
                self.status_var.set("车牌检测完成，但未加载识别模型")
        
        except Exception as e:
            print(f"识别车牌时出错: {e}")
            traceback.print_exc()
            self.status_var.set("识别过程出错")
            messagebox.showerror("错误", f"识别车牌时出错: {str(e)}")
    
    def _draw_detection_results(self):
        """在原始图像上绘制检测结果"""
        if self.input_image_cv is None or not self.plate_boxes:
            return
        
        # 复制原始图像，避免修改原始数据
        image_with_boxes = self.input_image_cv.copy()
        
        # 绘制所有检测到的车牌框
        for box in self.plate_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, "License Plate", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # BGR转RGB用于显示
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(image_with_boxes_rgb)
        
        # 调整大小以适应显示区域
        height, width = image_with_boxes_rgb.shape[:2]
        max_size = 560
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        display_annotated = annotated_image.resize((new_width, new_height), Image.LANCZOS)
        self.input_tk = ImageTk.PhotoImage(display_annotated)
        
        # 更新显示
        self.input_image_canvas.delete("all")
        self.input_image_canvas.config(width=new_width, height=new_height)
        self.input_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.input_tk)
    
    def _recognize_plate_text(self, plate_tensor):
        """识别车牌文本并显示中间过程可视化"""
        with torch.no_grad():
            # 前向传播
            logits, deblurred, theta, input_image, stn_output, feature_maps = self.lpr_model(
                plate_tensor, plate_tensor)
            log_probs = F.log_softmax(logits, dim=2)
            
            # 解码预测结果
            _, predictions = log_probs.max(2)
            predictions = predictions.cpu().numpy()[0]
            
            # CTC解码（简单贪婪解码）
            decoded_pred = []
            prev = -1
            for p in predictions:
                if p != 0 and p != prev:  # 不是blank且不重复
                    decoded_pred.append(p)
                prev = p
            
            # 转换为文本
            plate_number = ''.join([self.idx_to_chars.get(idx, '-') for idx in decoded_pred])
            
            # 创建可视化图像
            self._visualize_intermediate_results(
                input_image, stn_output, deblurred, feature_maps, plate_number)
            
            return plate_number
    
    def _visualize_intermediate_results(self, input_image, stn_output, deblurred, feature_maps, plate_number):
        """可视化中间处理结果"""
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        plt.subplot(3, 2, 1)
        self._show_tensor_image(input_image[0], "原始车牌图像")
        
        plt.subplot(3, 2, 2)
        if stn_output is not None:
            self._show_tensor_image(stn_output[0], "STN校正后图像")
        else:
            plt.text(0.5, 0.5, "STN未启用", ha='center', va='center')
            plt.axis('off')
        
        plt.subplot(3, 2, 3)
        if deblurred is not None:
            self._show_tensor_image(deblurred[0], "去模糊后图像")
        else:
            plt.text(0.5, 0.5, "去模糊未启用", ha='center', va='center')
            plt.axis('off')
        
        # 显示特征图
        plt.subplot(3, 2, 4)
        if feature_maps is not None:
            # 选择前16个通道的特征图进行可视化
            features = feature_maps[0].cpu().numpy()
            n_features = min(16, features.shape[0])
            
            # 创建一个复合特征图
            grid_size = int(np.ceil(np.sqrt(n_features)))
            feature_grid = np.zeros((grid_size * features.shape[1], grid_size * features.shape[2]))
            
            for i in range(n_features):
                row = i // grid_size
                col = i % grid_size
                feature_grid[row * features.shape[1]:(row + 1) * features.shape[1],
                            col * features.shape[2]:(col + 1) * features.shape[2]] = features[i]
            
            plt.imshow(feature_grid, cmap='viridis')
            plt.title("特征图")
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, "无特征图", ha='center', va='center')
            plt.axis('off')
        
        # 添加识别结果
        plt.subplot(3, 1, 3)
        plt.text(0.5, 0.5, f"识别结果: {plate_number}", ha='center', va='center', fontsize=20)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _show_tensor_image(self, tensor, title):
        """将张量转换为可显示的图像"""
        # 反归一化
        img = tensor.cpu().detach().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

# 程序入口
def main():
    root = tk.Tk()
    app = LicensePlateRecognitionSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()