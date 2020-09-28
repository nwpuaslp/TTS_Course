## 1. 实验要求

按照流程跑通g2p模型，计算测试集的准确率，并提交实验报告。

## 2. 实验步骤
### 2.1  数据

+ 共99164行数据。

+ 数据注音中^表示非中文字符。

+ 由于部分训练过程需要比较大的内存，如果你的计算机不支持全量数据的训练，你可以使用mini-train.dict来进行训练。

### 2.2 作业内容

1. [必做] 多音字n-gram模型

    + 编写脚本修改成训练需要的数据格式，格式请参考reference.txt.

    + 参照参考流程，训练提供的中文数据集，并在test.dict上做解码,使用测试脚本计算准确率。

2. [选做] 多音字rnnlm模型

    + 在完成实验1的情况下，参照参考流程，按照你在实验1中的数据格式训练，并在test.dict上做解码,使用测试脚本计算准确率。

   

   *备注:你可以通过调整部分参数来使模型性能提升。*

## 3 文件路径说明

+ dataset
  + train.dict 全量训练集
  + mini-train.dict 部分训练集
  + test.dict 测试集不带注音
  + gt.dict 测试集带注音
+ acc.py 准确率计算脚本。你的解码格式需要与gt.dict一致。
+ README.md 说明文件
+ reference.txt 需要的数据格式参考

## 4 参考流程

1. **[必做] 多音字n-gram模型**

+ This build was tested via AWS EC2 with a fresh Ubuntu 14.04 and 16.04 base, and m4.large instance.

```shell
$ sudo apt-get update
# Basics
$ sudo apt-get install git g++ autoconf-archive make libtool
# Python bindings
$ sudo apt-get install python-setuptools python-dev
# mitlm (to build a quick play model)
$ sudo apt-get install gfortran
```

+ Next grab and install OpenFst-1.6.2 (10m-15m):

```shell
$ wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz
$ tar -xvzf openfst-1.6.2.tar.gz
$ cd openfst-1.6.2
# Minimal configure, compatible with current defaults for Kaldi
$ ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
$ make -j 4
# Now wait a while...
$ sudo make install
$ cd
# Extend your LD_LIBRARY_PATH .bashrc:
$ echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst' \
     >> ~/.bashrc
$ source ~/.bashrc
```

+ Checkout the latest Phonetisaurus from master：

```shell
$ git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
$ cd Phonetisaurus
$ ./configure
$ make
$ sudo make install
```

+ Grab and install mitlm：

```shell
$ git clone https://github.com/mitlm/mitlm.git
$ cd mitlm/
$ ./autogen.sh
$ make
$ sudo make install
$ cd
```

+ Train an n-gram model：

```shell
$ cd Phonetisaurus
# train.corpus is the correct data format converted from train.dict, Please refer to reference.txt for the format.
$ estimate-ngram -o 8 -t train.corpus \
  -wl dict.o8.arpa
$ phonetisaurus-arpa2wfst --lm=dict.o8.arpa --ofile=dict.o8.fst
```

+ Generate pronunciations for test data using the wrapper script:

```shell
$ phonetisaurus-apply --model train/dict.o8.fst --word_list test.dict > yourresult
```



**[选做] 多音字rnnlm模型**

+ You should install OpenFst-1.5.0：

```shell
$ wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.5.0.tar.gz
$ tar -xvzf openfst-1.5.0.tar.gz
$ cd openfst-1.5.0
# Minimal configure, compatible with current defaults for Kaldi
$ ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
$ make -j 4
# Now wait a while...
$ sudo make install
$ cd
# Extend your LD_LIBRARY_PATH .bashrc:
$ echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst' \
     >> ~/.bashrc
$ source ~/.bashrc
```

+ Clone the RnnLMG2P：

```shell
$ git clone https://github.com/AdolfVonKleist/RnnLMG2P.git
$ cd RnnLMG2P
$ make && make install
```

+ Train the rnnlm model：

```shell
$ cd script/
# train.corpus is the correct data format converted from train.dict, Please refer to reference.txt for the format.
$ ./train-g2p-rnnlm.py -c train.corpus -p yourmodel
```

+ Generate pronunciations for test data:

```shell
$ ../phonetisaurus-g2prnn --rnnlm=test.rnnlm --test=test.dict --nbest=5 | ./prettify.pl > tmp.txt
$ awk -F '\t' '{print $1"\t"$2}' tmp.txt > yourresult
```



## 5 提交内容

使用训练好的模型执行以下命令：

```shell
python acc.py --src_path yourresult --gt_path gt_path
```

分别提交模型在测试集上的准确率文件acc.txt.
