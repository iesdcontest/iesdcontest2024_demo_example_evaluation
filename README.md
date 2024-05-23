# 测试代码说明
最终上板测试的代码模板请参考`/tensorflow/lite/micro/examples/af_detection/main_functions.cc`文件，其中第255-257行为AF检测函数接口。使用tflite实现检测的各参赛队伍仅需要对`./tflite-micro/tensorflow/lite/micro/models/`中的`.tflite`文件进行替换即可；用其他方式实现检测的队伍可对第255-257行内容根据自己的设计进行替换.

除此之外此文件中的的所有代码不可更改，或在更改后仍能够与上位机测试程序正常通讯和测试。

上位机测试程序请参考`evaluation_af_detection.py`

# 测试指标说明
**模型精度**、**模型泛化性**、**推理延时**三个最终测试指标均通过此测试repo得出。

**存储占用**以各参赛队伍最终提交的可执行文件大小为准。
