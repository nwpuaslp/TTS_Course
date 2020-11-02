# 深蓝学院作业 

## world 声码器

WORLD对应以下三种声学特征：F0基频、SP频谱包络与AP非周期序列。

正弦波组成的原始信号中频率最低的正弦波即为基频；

频谱包络是指将不同频率的振幅最高点通过平滑的曲线连接起来得到的包络线；

非周期序列对应混合激励部分的非周期脉冲序列。

### 用法

### compile

```bash
cd tools
bash compile_tools.sh
```

编译过程可能较久，请耐心等待，编译后得到 tools/bin 文件夹

### copy synthesis

根据输入 test.wav 通过 world声码器 合成 syn.wav

使用world提取test.wav的f0、sp、ap，然后根据提取出的特征合成 copy_synthesize/16k_wav_syn/000001.resyn.wav

#### 采样率16k

```bash
bash copy_synthesize/copy_synthesis_world_16k.sh
```

#### 采样率48k

在copy_synthesis_world_16k.sh基础上修改，可修改参数为输入输出路径(wav_dir\out_dir)、采样率fs、mcsize等

### (选做) melspectrogram  copy synthesis

根据输入 test.wav 通过 griffinlim 合成 syn_mel.wav

使用world提取test.wav的melspectrogram，然后根据提取出的melspectrogram特征合成 copy_synthesize/syn_mel.wav

```bash
python copy_synthesize/copy_synthesis_mel.py copy_synthesize/16k_wav/000001.wav
```

## 传统TTS模型 (时长模型、声学模型)

文本特征 -> 时长模型(duration model) -> 声学模型(acoustic model) -> World vocoder -> 音频

### 1、下载数据、配置环境

#### 下载数据

数据链接：https://pan.baidu.com/s/1_zN-PSIIrxtCGvjo1TWz1g

提取码：jbc5

解压得到 train_data 文件夹

数据链接：https://pan.baidu.com/s/1i28ZppgWYHIupk8piH7SyQ

提取码：mvkj

解压得到 test_data 文件夹

train_data/test_data

-----| acoustic_features    声学模型输入

-----| acoustic_targets     声学模型输出

-----| duration_features    时长模型输入

-----| duration_targets     时长模型输出

#### 配置环境(Python 3.6  Tensorflow 1.12)

安装python 包

```bash
pip install -r requirements.txt
```

安装较慢的可以使用whl安装

tensorflow 1.12 whl包：https://pan.baidu.com/s/1WCOyFhszJnHHtIMWBq0sxQ

提取码：r73i

### 2、归一化数据

对duration、acoustic输入输出数据进行归一化，得到cmvn中的train_cmvn_dur.npz和train_cmvn_spss.npz文件。

```bash
bash bash/prepare.sh
```

### 3、编写模型

完成 model.py 中 AcousticModel 和 DurationModel 模型定义部分

模型输入为 inputs 和 input_length, 预测结果为 targets

inputs为输入特征，input_length 为inputs的第一维维度， targets为预测结果

#### 输入输出

inputs.shape = [seq_length, feature_dim]

input_length = seq_length

target.shape = [seq_length, target_dim]

时长模型的input的feature_dim为617维，表示文本特征

时长模型的output的target_dim为5维，表示每个音素的状态时长信息

声学模型的input的feature_dim为626维，表示文本特征和帧的位置特征

声学模型的output的target_dim为75维，表示目标音频的声学特征(lf0,mgc,bap)

#### 任务说明

编写模型，根据输入inputs预测输出targets

### 4、训练

训练时长模型
```bash
bash bash/train_dur.sh
```

训练声学模型
```bash
bash bash/train_acoustic.sh
```

训练的模型结果分别保存在logdir_dur 和logdir_acoustic中

训练总步数,checkpoint保存步数等可以在hparams.py中修改，判断模型是否收敛可以通过loss曲线和测试合成的效果判断，不一定需要跑完总步数。

#### 通过tensorborad查看loss函数曲线

```bash
tensorboard --logdir=logdir_dur
```

打开浏览器查看本地6006端口

### 5、测试合成


时长模型的预测输出在 output_dur (作为声学模型的输入)

声学模型的预测结果在 output_acoustic (lf0、bap、mgc文件夹中为生成的对应特征，根据这些特征通过world声码器合成syn_wav中的音频)

\<checkpoint> 填入训练得到的模型路径

#### 1、测试声学模型(使用真实时长数据)

测试脚本第一个参数为输入label路径，第二个参数为输出路径，第三个参数为训练好的模型路径(例如：logdir_acoustic/model/model.ckpt-2000)

```bash
bash bash/synthesize_acoustic.sh test_data/acoustic_features output_acoustic <checkpoint>
```

#### 2、测试时长模型 + 声学模型(使用时长模型的输出作为声学模型的输入)

```bash
bash bash/synthesize_dur.sh test_data/duration_features output_dur <checkpoint>
bash bash/synthesize_acoustic.sh output_dur output_acoustic <checkpoint>
```

最终合成语音在output_acoustic/syn_wav中


## 上交作业

需要提交16k、48k采样率的copy synthesis结果

传统模型合成结果
