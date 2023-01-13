# 飞桨框架 MLU 版 DeepLabV3

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.DeepLabV3 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载并安装 PaddleSeg 套件

```bash
cd path_to_clone_PaddleSeg
# 下载套件代码的 develop 分支
git clone https://github.com/PaddlePaddle/PaddleSeg.git -b develop
# 如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：
git clone https://gitee.com/paddlepaddle/PaddleSeg.git -b develop
# 安装Python依赖库 - PaddleSeg 的 Python 依赖库在 requirements.txt 中给出
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# 安装发布的PaddleSeg包
pip install paddleseg
```
也可以访问PaddleSeg的 [Github Repo](https://github.com/PaddlePaddle/PaddleSeg.git) 直接下载源码。

### 1.3 准备训练数据集
```bash
# 准备数据集 - 将 VOC2012 放到对应的数据集目录下
cd PaddleSeg/data/VOCdevkit/
# 解压完成之后，数据集目录结构如下
PaddleSeg/data
└── VOCdevkit
 └── VOC2012
 ├── Annotations
 ├── ImageSets
 ├── JPEGImages
 ├── SegmentationClass
 └── SegmentationObject
# 运行下列命令进行SBD数据集进行扩充，其中voc_path应根据实际数据集路径进行设置
cd PaddleSeg
python tools/data/voc_augment.py --voc_path data/VOCdevkit --num_workers 8
# 扩充后的数据集目录结构如下
PaddleSeg/data
└── VOCdevkit
 └── VOC2012
 ├── Annotations
 ├── ImageSets
 ├── JPEGImages
 ├── SegmentationClass
 ├── SegmentationClassAug
 └── SegmentationObject
```

### 1.4 运行训练

使用飞浆 PaddleSeg 套件运行 MLU , 可以通过设置 device 参数为 mlu 来指定设备。   
```bash
# 进入模型目录
cd PaddleSeg
# 输出环境变量，使用0,1,2,3号卡
export MLU_VISIBLE_DEVICES=0,1,2,3

# 单卡训练 - 输出位于output/deeplabV3/
python train.py --config configs/deeplabv3/deeplabv3_resnet50_os8_voc12aug_512x512_40k.yml \
                --device mlu \
                --do_eval

# 四卡训练 - 输出位于output/deeplabV3/
python -m paddle.distributed.launch --mlus="0,1,2,3" train.py \
        --config configs/deeplabv3/deeplabv3_resnet50_os8_voc12aug_512x512_40k.yml \
        --device mlu \
        --do_eval
```
## 2.DeepLabV3 精度
| Model | dataset |mIOU FP32| mIOU AMP | 
| ------------- |------------- |------------- | ------------- |
| DeepLabV3 | Pascal VOC2012 + Aug | 79.76% | 79.46% |

## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleSeg模型套件下载链接：[https://github.com/PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- VOC2012数据集下载链接：[http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

## 4.Release_Notes
@TODO