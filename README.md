# 爱工千古（AIGongQianGu）项目 v1.0

AIGongQianGu 项目目前包含两个主要功能模块：**人脸识别** 和 **视频获取与切分**。本项目致力于为图像和视频数据的采集、处理、分析提供一站式解决方案。

The AIGongQianGu project currently includes two main modules: **Face Recognition** and **Video Acquisition & Segmentation**. This project aims to provide an all-in-one solution for image and video data collection, processing, and analysis.

---

## 目录结构 / Directory Structure

```
face_rec/                # 人脸识别相关代码 / Face recognition code
├── build_model.py       # 训练人脸识别模型 / Train face recognition model
├── test_accuracy.py     # 测试模型准确率 / Test model accuracy
├── image_getter/        # 图像采集与训练数据构建 / Image collection & dataset builder
│   ├── ai_spider.py
│   └── training_data_builder.py
├── images/              # 图片数据 / Image data
│   ├── train/           # 训练集 / Training set
│   └── test/            # 测试集 / Test set
├── img/                 # 其他图片资源 / Other image resources
└── model/
    └── face_model.pkl   # 已训练模型 / Trained model

video_process/           # 视频处理相关代码 / Video processing code
├── cut_by_second.py     # 按秒切割视频 / Cut video by seconds
├── video_getter.py      # 视频采集 / Video acquisition
├── files/               # 中间文件 / Intermediate files
└── videos/              # 视频数据 / Video data

dlib-19.24.2-cp312-cp312-win_amd64.whl  # dlib库本地安装包 / Local dlib wheel
```

---

## 安装与环境依赖 / Installation & Dependencies

- Python 3.12
- dlib 19.24.2
- 其他依赖请根据实际代码补充（如 numpy, opencv-python, pillow 等）

Install dlib (use local wheel if offline):
```bash
pip install dlib-19.24.2-cp312-cp312-win_amd64.whl
```

---

## 使用说明 / Usage

### 1. 人脸识别 / Face Recognition

- **训练模型 / Train Model**  
  运行 `face_rec/build_model.py`，使用 `images/train/` 下的图片进行模型训练，模型将保存在 `face_rec/model/face_model.pkl`。
  
  Run `face_rec/build_model.py` to train the model using images in `images/train/`. The trained model will be saved as `face_rec/model/face_model.pkl`.

- **测试准确率 / Test Accuracy**  
  运行 `face_rec/test_accuracy.py`，评估模型在 `images/test/` 上的表现。
  
  Run `face_rec/test_accuracy.py` to evaluate the model on `images/test/`.

### 2. 视频获取与切分 / Video Acquisition & Segmentation

- **视频采集 / Video Acquisition**  
  运行 `video_process/video_getter.py`，采集视频并保存到 `video_process/videos/`。
  
  Run `video_process/video_getter.py` to acquire videos and save them to `video_process/videos/`.

- **视频切割 / Video Segmentation**  
  运行 `video_process/cut_by_second.py`，可按秒切割视频片段。
  
  Run `video_process/cut_by_second.py` to segment videos by seconds.

---

## 后续开发计划 / Future Plans

- 视频字幕OCR识别（subtitle_orc.py）
- 搜索网站功能
- 更多智能分析与可视化

Planned features:
- Video subtitle OCR (subtitle_orc.py)
- Search website
- More intelligent analysis and visualization

---

## 版本 / Version

v1.0