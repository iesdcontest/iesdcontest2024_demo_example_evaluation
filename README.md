# 测试代码说明
最终上板测试的代码模板请参考`/tensorflow/lite/micro/examples/af_detection/main_functions.cc`文件，其中第254行`aiRun((void*)input,(void*)result)`为AF检测函数接口，各参赛队伍可对函数内容根据自己的设计进行替换.

除此之外此文件中的的所有代码不可更改，或在更改后仍能够与上位机测试程序正常通讯和测试。

上位机测试程序请参考`evaluation_af_detection.py`

# 测试指标说明
**模型精度**、**模型泛化性**、**推理延时**三个最终测试指标均通过此测试repo得出。

**存储占用**以各参赛队伍最终提交的可执行文件大小为准。
