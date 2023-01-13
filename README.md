
# 飞桨框架寒武纪 MLU 版支持模型

目前 Paddle MLU 版基于寒武纪 MLU370 系列板卡支持以下模型的单机单卡/单机多卡的训练。


本页面展示 PaddlePaddle-MLU 支持的网络列表，飞桨官方通过套件的方式对各种 AI 任务 （图像分类、目标检测、图像分割、自然语言处理、字符识别等）提供训练、推理的运行工具。开发者可在各个套件下通过工具脚本 + 配置文件的方式来执行具体的 AI 任务，详细方法可参考表格中模型使用一列。


## 图像分类

| 模型               | 领域     | 模型使用                                                   | 编程范式      |  训练单机多卡支持  | 
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- |
| ResNet50  | 图像分类 | [模型使用](./example/classification/ResNet50.md) |  动态图  | 支持 |
| VGG16/19 | 图像分类 | [模型使用](./example/classification/VGG16_VGG19.md) |  动态图  | 支持 | 
| InceptionV4 | 图像分类 | [模型使用](./example/classification/InceptionV4.md) |  动态图  | 支持 | 
| MobileNetV3 | 图像分类 | [模型使用](./example/classification/MobileNetV3.md) |  动态图  | 支持 | 


## 目标检测

| 模型               | 领域     | 模型使用                                                   | 编程范式      |  训练单机多卡支持  | 
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- |
| YOLOv3  | 目标检测 | [模型使用](./example/detection/YOLOV3.md) |  动态图  | 支持 |
| PP-YOLO | 目标检测 | [模型使用](./example/detection/PP-YOLO.md) |  动态图  | 支持 | 
| SSD | 目标检测 | [模型使用](./example/detection/SSD.md) |  动态图  | 支持 | 支持 | 
| Mask R-CNN  | 目标检测 | [模型使用](./example/detection/MaskRCNN.md) |  动态图  | 支持 | 
| Mask R-CNN + FPN  | 目标检测 | [模型使用](./example/detection/MaskRCNN_FPN.md) |  动态图  | 支持 |


## 图像分割

| 模型               | 领域     | 模型使用                                                   | 编程范式      |  训练单机多卡支持  | 
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- |
| DeepLabV3 | 图像分割 | [模型使用](./example/segmentation/DeepLabV3.md) |  动态图  | 支持 | 
| U-Net | 图像分割 | [模型使用](./example/segmentation/UNet.md) |  动态图  | 支持 |

## 自然语言处理

| 模型               | 领域     | 模型使用                                                   | 编程范式      |  训练单机多卡支持  | 
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- |
| BERT | NLP | [模型使用](./example/nlp/BERT.md) |  动态图  | 支持 |
| Transformer | NLP | [模型使用](./example/nlp/Transformer.md) |  动态图  | 支持 |


## 字符识别

| 模型               | 领域     | 模型使用                                                   | 编程范式      |  训练单机多卡支持  |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- |
| OCR-DB | 文本检测 | [模型使用](./example/ocr/OCR-DB.md) |  动态图  | 支持 |
| OCR-Clas | 角度分类 | [模型使用](./example/ocr/OCR-CLAS.md) |  动态图  | 支持 | 

## issues/wiki/forum 跳转链接

## contrib 指引和链接

## LICENSE

ModelZoo Paddle 的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 ModelZoo, ModelZoo 也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。
