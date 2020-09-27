## 1. 实验要求
按照流程跑通CRF分词模型，自己编写脚本计算测试集的相关指标（精确率、召回率，F1），并提交实验报告。
## 2. 实验步骤
### 2.1 编译CRF++
```bash
cd tools/
tar -zxvf CRF++-0.58.tar.gz
cd CRF++-0.58
./configure --prefix=$PWD/build
make
make install
export PATH=CRF++-0.58/build/bin/:$PATH
```
### 2.2 数据准备
文件夹中提供了来自人民日报的两份不同大小的数据集。读者根据自己设备的情况选择使用的数据集。

* 数据一: data/people_daliy_10W
* 数据二: data/people_daliy_26W

将数据预处理为CRF++ 模型的输入格式，可参照data/demo/train.data 的格式。下面以data/demo/中的数据说明整个流程。

```python
# 准备训练数据
python scripts/1_prepare_train.py data/demo/train.raw data/demo/train.data
# 准备测试数据
python scripts/2_prepare_test.py data/demo/test.raw data/demo/test.input
```

### 2.3 模型训练
```bash
crf_learn -f 3 -c 4.0 config/template data/demo/train.data model.demo
```

### 2.4 模型测试

#### 解码:
```
crf_test -m model.demo data/demo/test.input > data/demo/test.out
```
得到的test.out 即为解码结果。读者根据data/demo/test.raw 和 data/demo/test.out 计算精确率，召回率以及F1值。
