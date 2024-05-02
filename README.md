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



第一步：上位机 启动如下代码所示python文件
```python
import argparse
import serial
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


def main():
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    List = []
    resultList = []
    labelList = []
    timeList = []
    port = args.com  # set port number
    ser = serial.Serial(port=port, baudrate=args.baudrate, timeout=1)  # open the serial
    print(ser)
    f = open('./testset_index.txt', 'r')
    for line in f:
        List.append(line)

    for idx in tqdm(range(0, 2)):
        labelList.append(List[idx].split(',')[0])
        # load data from txt files and reshape to (1, 1, 1250, 1)
        testX = txt_to_numpy(args.path_data + List[idx].split(',')[1].strip(), 1250).reshape(1, 1, 1250, 1)
        # receive messages from serial port, the length is the number of bytes remaining in the input buffer
        for i in range(0, testX.shape[0]):
            # don't continue running the code until a "begin" is received, otherwise receive iteratively
            nus =0
            while ser.in_waiting < 5:
                time.sleep(1)
                # print("w1")
                nus = nus +1
                if nus >15 :
                    recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                    print(recv)
                    exit(1)
                pass
            # when receiving the code "begin", send the test data cyclically
            recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
            print(recv)
            # clear the input buffer
            ser.reset_input_buffer()
            #if True:
            if recv.strip() == 'begin':
                for j in range(0, testX.shape[1]):
                    for k in range(0, testX.shape[2]):
                        for l in range(0, testX.shape[3]):
                            time.sleep(0.003)
                            send_str = str(testX[i][j][k][l]) + ' '
                            ser.write(send_str.encode(encoding='utf8'))
                            
                # don't continue running the code until a "ok" is received
                while ser.in_waiting < 2:
                    time.sleep(0.01)
                    pass
                time.sleep(0.01)
                recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                print(recv)
                ser.reset_input_buffer()
                if recv.strip() == 'ok':
                    time.sleep(0.02)
                    # send status 200 to the board
                    send_str = '200 '
                    ser.write(send_str.encode(encoding='utf8'))
                    time.sleep(0.01)
                # receive results from the board, which is a string separated by commas
                while ser.in_waiting < 4:
                    pass
                recv = ser.read(ser.in_waiting).decode(encoding='utf8')
                print(recv)
                ser.reset_input_buffer()
                # the format of recv is ['<result>','<dutation>']
                result = recv.split(',')[0]
                inference_latency = recv.split(',')[1]
                if result == '0':
                    resultList.append('0')
                else:
                    resultList.append('1')
                # inference latency in ms
                timeList.append(float(inference_latency) /1000)

    total_time = sum(timeList)
    avg_time = np.mean(timeList)
    print(avg_time)
    C_labelList = np.array(labelList).astype(int)
    C_resultList = np.array(resultList).astype(int)
    C = confusion_matrix(C_labelList, C_resultList, labels=[0, 1])
    print(C)
    print(labelList)
    print(resultList)

    acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
    precision = C[1][1] / (C[1][1] + C[0][1])
    sensitivity = C[1][1] / (C[1][1] + C[1][0])
    FP_rate = C[0][1] / (C[0][1] + C[0][0])
    PPV = C[1][1] / (C[1][1] + C[1][0])
    NPV = C[0][0] / (C[0][0] + C[0][1])
    F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)

    print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
          "\nF_beta_score: {},\ntotal_time: {}ms,\n average_time: {}ms".format(acc, precision, sensitivity, FP_rate, PPV,
                                                                           NPV, F1_score, F_beta_score,
                                                                           total_time, avg_time))

    f = open('./log/2pack_log_{}.txt'.format(t), 'a')
    f.write("Accuracy: {}\n".format(acc))
    f.write("Precision: {}\n".format(precision))
    f.write("Sensitivity: {}\n".format(sensitivity))
    f.write("FP_rate: {}\n".format(FP_rate))
    f.write("PPV: {}\n".format(PPV))
    f.write("NPV: {}\n".format(NPV))
    f.write("F1_Score: {}\n".format(F1_score))
    f.write("F_beta_Score: {}\n".format(F_beta_score))
    f.write("Total_Time: {}ms\n".format(total_time))
    f.write("Average_Time: {}ms\n\n".format(avg_time))
    f.write(str(C) + "\n\n")
    f.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # 选择好数据串口的COM端口
    argparser.add_argument('--com', type=str, default='com7')
    argparser.add_argument('--baudrate', type=int, default=115200)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    args = argparser.parse_args()
    main()
```


第二步：打开MobaXterm（上位机）启用相应端口执行
```commandline
./af_detection
```
注意:上位机python程序中的端口为数据串口，选择需要在运行之前确认，流程为 此电脑--管理--设备管理器--端口

# 参考链接
[1]【广东龙芯2K500先锋板试用体验】运行边缘AI框架——TFLM：https://bbs.elecfans.com/jishu_2330951_1_1.html

