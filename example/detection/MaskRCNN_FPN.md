# 飞桨框架 MLU 版 Mask R-CNN + FPN 

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

## 1.Mask R-CNN + FPN 训练

### 1.1 安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](../../install/paddle_install_cn.md) 进行安装或编译。


### 1.2 下载并安装 PaddleDetection 套件

```bash
cd path_to_clone_PaddleDetection
# 下载套件代码的 develop 分支
git clone https://github.com/PaddlePaddle/PaddleDetection.git -b develop
# 如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：
git clone https://gitee.com/paddlepaddle/PaddleDetection.git -b develop
# 安装Python依赖库 - PaddleDetection 的 Python 依赖库在 requirements.txt 中给出
pip install --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
## 编译安装paddledet
cd PaddleDetection
python setup.py install
```
也可以访问 PaddleDetection 的 [GitHub Repo](https://github.com/PaddlePaddle/PaddleDetection) 直接下载源码。

### 1.3 准备训练数据集
请根据[数据说明](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/data/PrepareDetDataSet.md)文档准备 COCO 数据集。
```bash
# 准备数据集 - 将 COCO2017 放到对应的数据集目录下
cd PaddleDetection/dataset/coco
# 解压完成之后，数据集目录结构如下
PaddleDetection/dataset/coco
├── annotations
│ ├── instances_train2017.json
│ ├── instances_val2017.json
│ | ...
├── train2017
│ ├── 000000000009.jpg
│ ├── 000000580008.jpg
│ | ...
├── val2017
│ ├── 000000000139.jpg
│ ├── 000000000285.jpg
│ | ...
| ...
```

### 1.4 运行训练

使用飞浆 PaddleDetection 套件运行 MLU，可以通过设置 use_mlu 参数为 mlu 来指定设备。
```bash
# 进入模型目录
cd PaddleDetection
# 设置环境变量，使用0,1,2,3,4,5,6,7号卡
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 单卡训练 - 输出位于output/mask_rcnn_r50_fpn_1x_coco/
python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
                      -o use_mlu=True use_gpu=False

# 八卡训练 - 输出位于output/mask_rcnn_r50_fpn_1x_coco/
python -m paddle.distributed.launch --mlus="0,1,2,3,4,5,6,7" tools/train.py \
          -c ./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
          -o use_mlu=True use_gpu=False

```
## 2.Mask R-CNN + FPN 精度
| Model | dataset |Box AP FP32| Segm AP FP32 |Box AP AMP| Segm AP AMP |
| ------------- |------------- |------------- | ------------- | ------------- | ------------- | 
| Mask R-CNN + FPN | COCO 2017 | 38.0 | 34.5 | 38.1 | 34.2 | 
## 3.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
- PaddleDetection模型套件下载链接：[https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- COCO数据集下载链接：[https://cocodataset.org/#download](https://cocodataset.org/#download)

## 4.Release_Notes
@TODO
