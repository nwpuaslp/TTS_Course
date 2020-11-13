# WaveRNN
## 1.实验要求
1. 利用db_1数据训练基于WaveRNN模型的neural vocoder，并补全解码部分的代码用于测试合成效果。
2. 与之前实验课程中训练的acoustic model进行对接，对结果进行评估。
3. 有余力的同学可以对UpsampleNet或是WaveRNN本身结构进行改动已验证不同效果。

## 2.实验步骤
### 2.1 数据准备
和之前实验课程一致，我们仍然使用基于标贝科技开源的10小时中文数据，下载链接：

```
链接: https://pan.baidu.com/s/1xC0fNwDvJWpJdqnfqNAzMg  密码: tgtp
```

数据准备流程请参照第四次课程（04_seq2seq_tts/README.md），可直接利用之前课程提取好的特征进行vocoder训练。

### 2.2 模型训练
首先执行```pip install -r requirement.txt```安装所需要环境。由于该网络参数量较大，建议使用带有GPU的服务器进行训练测试。另外，由于同学们的cuda版本等并非一致，可根据自己的实际情况对pytorch的版本进行更改，详见 https://pytorch.org/

将按照之前课程准备好的数据集作为参数传递给训练脚本
```
python train.py --data_dir db_1/
```

### 2.3 模型解码
按照给出的提示，补全WaveRNN/layers/wavernn.py中的解码逻辑。

合成验证集中的真实mel谱特征
```
python generate.py --data_dir db_1/
```

之后，在和前次合成声学模型预测的mel谱对接时，需改动generate.py中数据读取的代码，或按照db_1的结构组织目录
