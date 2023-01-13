# 飞桨框架 MLU 版 OCR-Clas

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.OCR-Clas 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载并安装 PaddleOCR 套件

```bash
cd path_to_clone_PaddleOCR
# 下载套件代码的 develop 分支
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：
git clone https://gitee.com/paddlepaddle/PaddleOCR.git
# 安装Python依赖库 - PaddleOCR 的 Python 依赖库在 requirements.txt 中给出
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
也可以访问PaddleOCR的 [Github Repo](https://github.com/PaddlePaddle/PaddleOCR.git) 直接下载源码。

### 1.3 准备训练数据集
```bash
# 准备自定义数据集
# 最终数据集应有如下文件结构：
├── train_data
│   └── cls
│       ├── cls_gt_test.txt
│       ├── cls_gt_train.txt
│       ├── test
│       │   ├── word_0.png
│       │   ├── word_1000.png
│       │   ├── word_1001.png
│       │   ├── word_1002.png
|       |   |——...
│       ├── train
│       │   ├── word_0.png
│       │   ├── word_1000.png
│       │   ├── word_1001.png
│       │   ├── word_1002.png
|       |   |——...
```
若您本地没有数据集，可以在官网下载 [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) 数据，用于快速验证。   
下载好数据集后，需要手动删除明显旋转 90 度的图片。生成文本方向为 0 和 180 度的图片，参考下面的python脚本对数据进行处理：
```python
from math import *
import cv2
import os
import glob
import numpy as np
from pathlib import Path
import random
def rotate_img(img, angle):
    '''
    img --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img
if __name__ == '__main__':
    output_dirs = ['train_data/cls/train','train_data/cls/test']
    files_output = ['train_data/cls/cls_gt_train.txt','train_data/cls/cls_gt_test.txt']
    paths = ['train_data/rec/train','train_data/rec/test']
    angle = [0,180]
    for (path, filename_output, output_dir) in zip(paths, files_output, output_dirs):
        output = []
        num = 0
        lst = os.listdir(path)
        for ll in lst:
        i = random.randint(0,1)
        image = cv2.imread(os.path.join(path,ll), -1)
        ##方法1
        h, w = image.shape[:2]
        if (h > w):
            num +=1
            continue
        rotated_img1 = rotate_img(image, angle[i])
        cv2.imwrite(os.path.join(output_dir, 'word_'+str(num)+'.png'), rotated_img1)
        with open(filename_output, 'a') as out_file:
        out_file.write('word_'+str(num)+'.png' + '\t' + str(angle[i]) + '\n')
        num +=1
```
### 1.4 运行训练

使用飞浆 PaddleOCR 套件运行 MLU，可以通过设置 Global.use_mlu 参数为 mlu 来指定设备。   
```bash
# 进入模型目录
cd PaddleOCR
# 输出环境变量，使用0,1,2,3号卡
export MLU_VISIBLE_DEVICES=0,1,2,3

# 单卡训练 - 输出位于output/cls/mv3 
python tools/train.py -c configs/cls/cls_mv3.yml \
       -o Global.use_gpu=False \
       Global.use_mlu=True

# 四卡训练 - 输出位于output/cls/mv3
python -m paddle.distributed.launch --mlus="0,1,2,3" tools/train.py \
            -c configs/cls/cls_mv3.yml \
            -o Global.use_gpu=False \
            Global.use_mlu=True
```
## 2.OCR-Clas 精度
| Model | dataset |ACC FP32| ACC AMP |
| ------------- |------------- |------------- | ------------- | 
| OCR-Clas | ICDAR2015 | 96.8% | 96.8% |
## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleOCR模型套件下载链接：[https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- ICDAR2015数据集下载链接：[https://rrc.cvc.uab.es/](https://rrc.cvc.uab.es/)

## 4.Release_Notes
@TODO
