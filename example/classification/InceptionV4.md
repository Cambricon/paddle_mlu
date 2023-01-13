# 飞桨框架 MLU 版 InceptionV4 

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.InceptionV4 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载并安装 PaddleClas 套件

```bash
cd path_to_clone_PaddleClas
# 下载套件代码的 develop 分支
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
# 如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b develop
# 安装Python依赖库 - PaddleClas 的 Python 依赖库在 requirements.txt 中给出
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
也可以访问 PaddleClas 的 [GitHub Repo](https://github.com/PaddlePaddle/PaddleClas) 直接下载源码。

### 1.3 准备训练数据集
请根据[数据说明](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/data_preparation/classification_dataset.md)文档准备 ImageNet1k 数据集。
```bash
# 准备数据集 - 将 ImageNet1k 放到对应的数据集目录下
cd PaddleClas/dataset/ILSVRC2012/
# 解压完成之后，数据集目录结构如下
PaddleClas/dataset/ILSVRC2012
├── train
├── train_list.txt
├── val
└── val_list.txt
```

### 1.4 运行训练

使用飞浆 PaddleClass 套件运行 MLU ，可以通过设置 Global.device 参数为 mlu 来指定设备。

```bash
#进入模型目录
cd PaddleClas/

#输出环境变量，使用0,1,2,3号卡
export MLU_VISIBLE_DEVICES=0,1,2,3

#单卡训练 - 输出位于output/InceptionV4
python tools/train.py \
          -c ./ppcls/configs/ImageNet/Inception/InceptionV4.yaml \
          -o Global.device=mlu

#四卡训练 - 输出位于output/InceptionV4
python -m paddle.distributed.launch --mlus="0,1,2,3" tools/train.py \
          -c ./ppcls/configs/ImageNet/Inception/InceptionV4.yaml \
          -o Global.device=mlu
```
## 2.InceptionV4 精度
| Model | dataset |Top1 FP32| Top1 AMP | 
| ------------- | ------------- |------------- |------------- |
| InceptionV4 | ImageNet2012 |80.02% | 80.26% | 

## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleClas模型套件下载链接：[https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- ImageNet1k数据集下载链接：[https://image-net.org/challenges/LSVRC/2012/](https://image-net.org/challenges/LSVRC/2012/)

## 4.Release_Notes
@TODO
