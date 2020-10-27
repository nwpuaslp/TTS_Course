# 深蓝学院作业 

## 实现Tacotron中的CBHG编码器

基于注意力机制的Tacotron模型包括编码器和带注意力机制的解码器两部分，这一次的作业只需要实现基于CBHG的编码器部分。

### 数据

基于标贝科技开源的10小时中文数据，我们提供处理好了的文本特征，并将其和音频文件一起打包到了百度网盘，下载链接：

```
链接: https://pan.baidu.com/s/1xC0fNwDvJWpJdqnfqNAzMg  密码: tgtp
```

在tacotron目录下的testdata中，我们也直接提供了该数据库的部分数据来测试流程。

### 环境
首先执行```pip install -r requirement.txt```安装所需要环境，该环境也是本作业的测试环境。

由于该网络参数量较大，建议使用带有GPU的服务器进行训练测试。


### 流程

首先，在egs/example目录中，运行```bash preprocess.sh 4```，可以针对性testdata中的文本和语音数据进行特征提取。

特征提取完成后，执行 ```bash train.sh```就可以进行模型训练。但是此时会提示出错，错误出现的地方在models/basic_model.py中，该处就是我们需要实现Tacotron的CBHG编码器部分。

CBHG实现完成后，可直接进行训练。

训练完成后，仿照egs/example/synthesis.sh编写合成脚本测试。由于此时使用的注意力机制是代码讲解中提到的最原始的注意力机制，收敛性能和效果都较差。在下一次作业中我们会实现更稳定的Tacotron系统。
