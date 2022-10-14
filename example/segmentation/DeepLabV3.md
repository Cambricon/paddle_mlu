# 飞桨框架 MLU 版 DeepLabV3

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.DeepLabV3 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载 PaddleSeg 代码，并准备 Pascal VOC 2012 + Aug 数据集

```bash
cd path_to_clone_PaddleSeg
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

### 1.3 运行训练

使用飞浆 PaddleSeg 套件运行 MLU , 可以通过设置 device 参数为 mlu 来指定设备。
```bash
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --mlus 0,1,2,3,4,5,6,7 train.py \
          --config configs/deeplabv3/deeplabv3_resnet50_os8_voc12aug_512x512_40k.yml \
          --device mlu \
          --do_eval
```
## 2.DeepLabV3 精度
| Model | dataset |mIOU FP32| mIOU AMP | 
| ------------- |------------- |------------- | ------------- |
| DeepLabV3 | CoCo 2017 | 79.76% | 79.46% |

## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleSeg模型套件下载链接：https://github.com/PaddlePaddle/PaddleSeg
- VOC2012数据集下载链接：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

## 4.Release_Notes
@TODO