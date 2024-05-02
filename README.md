# 龙芯先锋板运行端侧AI

本次大赛提供龙芯先锋板运行端侧AI的基础例程，参赛队伍也可根据自身需求，选择不同深度学习框架完成预训练模型在先锋板上的部署和推理计算。


# 一、交叉编译（PC端）

通过大赛提供的[Tensorflow例程](https://github.com/iesdcontest/iesdcontest2024_demo_example_tensorflow.git),
在训练数据集训练后，可以获得`.tflite`格式的模型权重文件。
本例程基于[tflite-micro](https://github.com/tensorflow/tflite-micro)项目，描述如何将`.tflite`神经网络模型部署至龙芯2K500先锋板。


## 1.1 安装依赖的python环境

下载python3环境和pillow库，在Linux环境中运行如下命令（使用的环境是WSL Ubuntu 22.04.4）

```cpp
sudo apt install python3 git unzip wget build-essential
```

执行如下指令确认pillow库是否已经完成安装

```cpp
pip install pillow
```

如果安装过程报错：The headers or library files could not be found for jpeg

可以尝试安装libjepg库：

```cpp
apt-get install libjpeg-dev zlib1g-dev
```

## 1.2  基准测试命令

首先下载[iesdcontest2024_demo_example_deployment](https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment)代码，使用如下命令：

```commandline
git clone https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment.git
```
将在服务器上训练完成的`.tflite`模型文件，更名为`af_detect.tflite`，放置于
```commandline
./tflite-micro/tensorflow/lite/micro/models/
```

## 1.3 环境配置

配置loongarch64-linux-gnu-gcc环境，解压龙芯交叉编译工具链的[压缩包](http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1.tar.xz)，将龙芯交叉编译工具链所在目录添加到PATH环境变量中；

```cpp
export PATH=$PATH:/PATH2GNU/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/
export ARCH=loongarch64
export CROSS_COMPILE=loongarch64-linux-gnu
```

配置成功后，可以使用如下命令进行测试：

```cpp
loongarch64-linux-gnu-gcc -v
```

## 1.4 交叉编译测试

进入`iesdcontest2024_demo_example_deployment`文件夹.
```commandline
cd iesdcontest2024_demo_example_deployment
```
在终端运行命令：
```cpp
MKFLAGS="-f tensorflow/lite/micro/tools/make/Makefile"
MKFLAGS="$MKFLAGS CC=loongarch64-linux-gnu-gcc"
MKFLAGS="$MKFLAGS CXX=loongarch64-linux-gnu-g++"
make  $MKFLAGS af_detection -j8
```

值得注意的是，如果此前使用的是sudo，则此处make语句也需要添加sudo，而loongarch环境需要在root用户下配置才有效，同时如果失效则可以尝试找到在root语句下配置的环境位置随后对上述语句做如下修改（此处以opt/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3/bin路径为示例）：

<!-- MKFLAGS="-f tensorflow/lite/micro/tools/make/Makefile" -->
<!-- MKFLAGS="$MKFLAGS CC=/opt/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3/bin/loongarch64-linux-gnu-gcc" -->
<!-- MKFLAGS="$MKFLAGS CXX=/opt/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3/bin/loongarch64-linux-gnu-g++" -->
<!-- make  $MKFLAGS run_af_detection -j8 -->
<!-- 通过上述命令可以实现对keyword_scrambled.tflite文件和person_detect.tflite文件，此处我们推荐只编译person_detect.tflite。 -->

于是将上述代码修改为：

```cpp
MKFLAGS="-f tensorflow/lite/micro/tools/make/Makefile"
MKFLAGS="$MKFLAGS CC=/opt/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3/bin/loongarch64-linux-gnu-gcc"
MKFLAGS="$MKFLAGS CXX=/opt/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3/bin/loongarch64-linux-gnu-g++"
make  $MKFLAGS af_detection -j8
```

运行成功后，会在`./gen/linux_x86_64_default_gcc/bin/`得到`af_detection`可执行文件。通过如下命令，可以确认生成的是LoongArch的可执行文件：
![](https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment/raw/main/img/af_detection_cross_compile.png)


# 二、神经网络部署（龙芯2K500先锋板）

## 2.1 上位机与先锋板的串口通信设置

此处通讯为串口方式（也可采用ssh）。因此，需要两根串口线连接先锋板与上位机（PC机）。接线方式如图所示：
![](https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment/raw/main/img/communication-2k500.png)

1. 针对debug串口接线：打开下载好的MobatXterm工具，点击Sesssion按键进入下一个页面，随后点击按键Serial,选择对应的COM口，设置波特率为115200，点击下方的Flow control选择None，其余按照默认值即可；
    具体接线可参考[龙芯2K500先锋板用户手册](https://1drv.ms/b/s!Aoaif3eONXLCdRdW6pvIiqclVNU?e=09lsKx)章节四-4.1。

2. 针对数据串口接线：接线方式如上图所示，USB端口与PC机连接。

## 2.2  神经网络推理计算的可执行文件的板上部署

可以通过使用U盘将第二步产生的可执行文件拷至2K500；U盘中文件考入2K500过程如下：

1、为U盘命名（挂载）
创建新文件
```
mkdir /mnt/usb/
```

```
mount /dev/sda1 /mnt/usb/
```
/dev/ U盘名 将U盘挂载在文件夹usb

U盘名字的查询
```
fdisk -l
```

2、使用挂载的名字将U盘中内容转入2K500
使用如下命令将文件复制到标记目录下（此处目录为用户根目录）：
```
cp /mnt/usb/person_detection ~/
``` 


## 2.3  神经网络推理计算与测试



第一步：上位机侧运行[evaluation_af_detection.py](https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment/blob/main/evaluation_af_detection.py)


第二步：打开MobaXterm（上位机）启用相应端口执行
```commandline
./af_detection
```
注意:上位机python程序中的端口为数据串口，选择需要在运行之前确认，流程为 此电脑--管理--设备管理器--端口

# 参考链接
[1]【广东龙芯2K500先锋板试用体验】运行边缘AI框架——TFLM：https://bbs.elecfans.com/jishu_2330951_1_1.html

