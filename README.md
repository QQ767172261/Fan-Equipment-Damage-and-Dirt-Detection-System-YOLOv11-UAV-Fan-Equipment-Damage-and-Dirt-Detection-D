风机设备损伤、脏污检测系统 YOLOv11 无人机-风机设备损伤、脏污检测数据集 无人机风力发电叶片缺陷检测数据集
无人机-风机设备损伤、脏污检测数据集，2191张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1ba7788679034cedb01c02507ae5a554.png)

图像尺寸:371*586
类别数量:2类
训练集图像数量:1534; 验证集图像数量:431； 测试集图像数量:226
类别名称: 每一类图像数 ，每一类标注数
damage: 1852,6380
dirt: 407,422
image num: 2191

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b976c6a5a4f04adb9ed3d3ecacc0e391.png)

模型代码
采用 YOLOv11n 网络训练
训练轮次：80 个 epoch
提供全部训练 + 测试源代码
训练精度 mAP 效果如图所示![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6a2515d72d34ab0bc5cfb63c57b09b4.png)

PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8db717b27e464dbca9ce24d902ae0d31.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测在这里插入图片描述

界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出
操作简单直观，无需命令行
4、
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81dab92eea2441189ceb7c8839fce343.png)

运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统

# 风机设备损伤、脏污检测系统 YOLOv11 全套构建代码
## 1. 数据集配置文件 `wind_turbine.yaml`
```yaml
path: ./wind_turbine_dataset
train: train/images
val: val/images
test: test/images

names:
  0: damage   # 风机损伤
  1: dirt     # 表面脏污
```

## 2. 训练脚本 `train.py`
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11n预训练权重
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="wind_turbine.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        pretrained=True,
        patience=10,
        save=True,
        conf=0.25
    )
    print("训练完成，最优权重：runs/detect/train/weights/best.pt")
```

## 3. PyQt5可视化界面完整代码 `main.py`（
```python
import sys
import cv2
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
                             QTableWidget, QTableWidgetItem, QComboBox, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO


class DetectThread(QThread):
    signal_img = pyqtSignal(QPixmap)
    signal_table = pyqtSignal(list)
    signal_info = pyqtSignal(float, int)

    def __init__(self, model, img_path, conf_thresh):
        super().__init__()
        self.model = model
        self.img_path = img_path
        self.conf_thresh = conf_thresh

    def run(self):
        t_start = time.time()
        res = self.model.predict(self.img_path, conf=self.conf_thresh)
        use_time = time.time() - t_start
        result_img = res[0].plot()
        target_num = len(res[0].boxes)
        info_list = []
        for box in res[0].boxes:
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy()
            info_list.append([cls_name, f"{conf:.2f}", int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
        h, w, c = result_img.shape
        qimg = QImage(result_img.data, w, h, w * c, QImage.Format_RGB888)
        self.signal_img.emit(QPixmap.fromImage(qimg))
        self.signal_table.emit(info_list)
        self.signal_info.emit(use_time, target_num)


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的风机损伤检测系统")
        self.resize(1400, 900)
        self.model = YOLO("runs/detect/train/weights/best.pt")
        self.initUI()

    def initUI(self):
        # 图像显示区
        self.label_img = QLabel("图像显示区域")
        self.label_img.setFixedSize(900, 700)
        self.label_img.setStyleSheet("border:1px solid #999;")

        # 右侧控制面板
        group_file = QGroupBox("文件导入")
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("摄像头未开启")
        layout_file = QVBoxLayout()
        layout_file.addWidget(self.btn_img)
        layout_file.addWidget(self.btn_video)
        layout_file.addWidget(self.btn_cam)
        group_file.setLayout(layout_file)

        group_result = QGroupBox("检测结果")
        self.label_time = QLabel("用时：0 s")
        self.label_count = QLabel("目标数目：0")
        self.combo_cls = QComboBox()
        self.combo_cls.addItem("全部")
        self.label_conf = QLabel("置信度阈值：0.87")
        self.label_pos = QLabel("xmin:0 ymin:0 | xmax:0 ymax:0")
        self.btn_save = QPushButton("保存")
        self.btn_exit = QPushButton("退出")
        layout_res = QVBoxLayout()
        layout_res.addWidget(self.label_time)
        layout_res.addWidget(self.label_count)
        layout_res.addWidget(self.combo_cls)
        layout_res.addWidget(self.label_conf)
        layout_res.addWidget(self.label_pos)
        layout_res.addWidget(self.btn_save)
        layout_res.addWidget(self.btn_exit)
        group_result.setLayout(layout_res)

        # 底部结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "xmin", "ymin", "xmax", "ymax"])

        # 整体布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(group_file)
        right_layout.addWidget(group_result)
        main_top = QHBoxLayout()
        main_top.addWidget(self.label_img)
        main_top.addLayout(right_layout)
        main_all = QVBoxLayout()
        main_all.addLayout(main_top)
        main_all.addWidget(self.table)
        central = QWidget()
        central.setLayout(main_all)
        self.setCentralWidget(central)

    def load_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png;*.jpeg")
        if not path:
            return
        self.thread = DetectThread(self.model, path, 0.25)
        self.thread.signal_img.connect(self.show_img)
        self.thread.signal_table.connect(self.fill_table)
        self.thread.signal_info.connect(self.show_info)
        self.thread.start()

    def show_img(self, pix):
        pix = pix.scaled(self.label_img.size(), Qt.KeepAspectRatio)
        self.label_img.setPixmap(pix)

    def show_info(self, t, num):
        self.label_time.setText(f"用时：{t:.3f} s")
        self.label_count.setText(f"目标数目：{num}")

    def fill_table(self, data):
        self.table.setRowCount(len(data))
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())
```

## 4. 环境安装命令
```bash
pip install ultralytics torch opencv-python pyqt5
```
## 系统功能
1. 图片、视频、摄像头实时检测风机叶片损伤、表面脏污两类目标
2. 界面展示推理耗时、目标数量、类别置信度、目标坐标
3. 检测结果表格展示，支持保存检测图片
4. 绘制蓝色检测框+类别+置信度标签，



