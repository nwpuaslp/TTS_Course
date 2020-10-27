## 1. 实验要求

按照流程跑通g2p模型，计算测试集的准确率，并提交实验pipeline。

## 2. 实验步骤
### 2.1  数据

+ 共99164行数据。

+ 数据注音中^表示非汉字字符,如标点符号等。

+ 由于部分训练过程需要比较大的内存，如果计算机不支持全量数据的训练，可以使用mini-train.dict来进行训练。

### 2.2 作业内容

1. ##### [必做] G2P n-gram模型

    + 4.1给出了开源英文数据集CMU的训练、测试等过程，读者可以先参照4.1流程，跑通英文数据集，在理清流程的情况下，训练提供的中文数据集，并在test.dict上做解码，使用测试脚本计算准确率。
    + 英文部分过程和中文不同，中文数据无法使用phonetisaurus-align。需要编写脚本将提供的中文数据修改成训练需要的数据格式，格式请参考reference.txt.

2. ##### [选做] G2P rnnlm模型

    + 在完成实验1的情况下，参照4.2流程，按照实验1中的数据格式训练，并在test.dict上做解码,使用测试脚本计算准确率。

 

***备注:读者可以通过调整部分参数来使模型性能提升***

## 3 文件路径说明

+ dataset
  + train.dict 全量训练集
  + mini-train.dict 部分训练集
  + test.dict 测试集不带注音(n-gram模型实验)
  + gt.dict 测试集带注音(n-gram模型实验)
  + testRnnlm.dict 测试集不带注音(Rnnlm模型实验)
  + gtRnnlm.dict 测试集带注音(Rnnlm模型实验)
+ acc.py 准确率计算脚本。解码格式需要与gt.dict一致。
+ README.md 说明文件
+ reference.txt 需要的数据格式参考

## 4 参考流程

#### 1.[必做] G2P n-gram模型

##### a. 安装相关工具与配置环境

+ This build was tested via AWS EC2 with a fresh Ubuntu 14.04 and 16.04 base, and m4.large instance.

```shell
$ sudo apt-get update
# Basics
$ sudo apt-get install git g++ autoconf-archive make libtool
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

+ Grab and install mitlm to build a quick test model with the cmudict (5m): 

```shell
$ git clone https://github.com/mitlm/mitlm.git
$ cd mitlm/
$ ./autogen.sh
$ make
$ sudo make install
$ cd
```

+ Grab a copy of the latest version of CMUdict and clean it up a bit:

```shell
$ mkdir example
$ cd example
$ wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
# Clean it up a bit and reformat:
$ cat cmudict.dict \
  | perl -pe 's/\([0-9]+\)//;
              s/\s+/ /g; s/^\s+//;
              s/\s+$//; @_ = split (/\s+/);
              $w = shift (@_);
              $_ = $w."\t".join (" ", @_)."\n";' \
  > cmudict.formatted.dict
```

##### b. 修改数据格式与训练

+ Align, estimate, and convert a joint n-gram model step-by-step:

```shell
# Align the dictionary (5m-10m)
$ phonetisaurus-align --input=cmudict.formatted.dict \
  --ofile=cmudict.formatted.corpus --seq1_del=false
# Train an n-gram model (5s-10s):
$ estimate-ngram -o 8 -t cmudict.formatted.corpus \
  -wl cmudict.o8.arpa
# Convert to OpenFst format (10s-20s):
$ phonetisaurus-arpa2wfst --lm=cmudict.o8.arpa --ofile=cmudict.o8.fst
$ cd
```

##### c. 解码

+ Generate pronunciations for test data using the wrapper script:

```shell
$ phonetisaurus-apply --model train/cmudict.o8.fst --word_list test.wlist > yourresult
```

#### 2.[选做] G2P rnnlm模型

##### a.  安装相关工具与配置环境

+ You should install OpenFst-1.5.0(高版本不支持)：

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
$ cd RnnLMG2P/src
$ make && make install
```

##### b. 训练

+ Train the rnnlm model：

```shell
$ cd script/
# train.corpus is the correct data format converted from train.dict, Please refer to reference.txt for the format. 
# You should modify "--classes" to meet the Chinese dataset's Pinyin classes. Here the classes are 1255.
# And you can also adjust the parameter like "--hidden" for better training. "--help" can help you know what parameters can be adjusted.
$ ./train-g2p-rnnlm.py -c train.corpus -p yourmodel  
```

##### c. 解码

+ Generate pronunciations for test data:

```shell
$ ../phonetisaurus-g2prnn --rnnlm=test.rnnlm --test=testRnnlm.dict --nbest=1 | ./prettify.pl > tmp.txt
$ awk -F '\t' '{print $1"\t"$2}' tmp.txt > yourresult
```



## 5 提交内容

##### 1.提交准确率

使用训练好的模型执行以下命令：

```shell
python acc.py --src_path yourresult --gt_path gt_path
```

分别提交模型在测试集上的准确率文件acc.txt.

##### 2.提交pipeline

分别提交中文数据集实验pipeline，需要准备一个shell脚本，这个脚本可以一次性跑通整个实验流程，并生成准确率，这其中不包括安装和环境配置过程，并且可以给脚本中的每一步写上注释。
