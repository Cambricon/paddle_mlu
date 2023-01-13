# 飞桨框架 MLU 版 OCR-DB 

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.OCR-DB 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载并安装 PaddleOCR 套件

```bash
cd path_to_clone_PaddleOCR
# 下载套件代码的dygraph分支
git clone https://github.com/PaddlePaddle/PaddleOCR.git 
# 如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：
git clone https://gitee.com/paddlepaddle/PaddleOCR.git 
# 安装Python依赖库 - PaddleOCR 的 Python 依赖库在 requirements.txt 中给出
pip install --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr # Recommend to use version 2.0.1+
## 编译安装paddleocr
cd PaddleOCR
python setup.py install
```
也可以访问PaddleOCR的 [Github Repo](https://github.com/PaddlePaddle/PaddleOCR) 直接下载源码。

### 1.3 准备训练数据集
```bash
# 准备数据集 - 将 ICDAR 2015 放到对应的数据集目录下
cd PaddleOCR
mkdir train_data
# 解压完成之后，将官网 label 转换为支持的数据格式。数据集目录结构如下：
PaddleOCR/train_data
└── icdar2015
 └── text_localization
 ├── ch4_test_images
 ├── icdar_c4_train_imgs
 ├── test_icdar2015_label.txt
 └── train_icdar2015_label.txt
```

### 1.4 运行训练

使用飞浆 PaddleOCR 套件运行 MLU，可以通过设置 Global.use_mlu 参数为 mlu 来指定设备。
```bash
# 进入模型目录
cd PaddleOCR
# 设置环境变量，使用0,1,2,3号卡
export MLU_VISIBLE_DEVICES=0,1,2,3

# 单卡训练 - 输出位于output/db_mv3/
python tools/train.py -c configs/det/det_mv3_db.yml \
        -o Global.use_mlu=True Global.use_gpu=False 

# 四卡训练 - 输出位于output/db_mv3/
python -m paddle.distributed.launch --mlus '0,1,2,3' tools/train.py \
          -c configs/det/det_mv3_db.yml \
          -o Global.use_gpu=False Global.use_mlu=True
```
## 2.OCR-DB 精度
| Model | dataset |PRECISION FP32| RECALL FP32 |HMEAN FP32|
| ------------- |------------- |------------- | ------------- | ------------- |
| OCR-DB  | ICDAR2015 | 80.90% | 69.3% | 74.7% |
## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleOCR模型套件下载链接：[https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- ICDAR2015数据集下载链接：[https://rrc.cvc.uab.es/](https://rrc.cvc.uab.es/)

## 4.Release_Notes
@TODO