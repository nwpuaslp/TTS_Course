# -*- coding:utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--src_path',
    default="./src.dict",
    help='your result path'
)
parser.add_argument(
    '--gt_path',
    default="./dataset/gt.dict",
    help='ground truth path'
)
args = parser.parse_args()

sum_line = 0
acc_line = 0
sum_num = 0
acc_num = 0

out_path = './acc.txt'

with open(args.src_path, 'r') as src_file:
    with open(args.gt_path, 'r') as gt_file:
        for src_line in src_file.readlines():
            sum_line += 1
            gt_line = gt_file.readline()
            src_yin = src_line.strip('\n').split('\t')[1]
            gt_yin = gt_line.strip('\n').split('\t')[1]
            if src_yin == gt_yin:
                acc_line += 1
            src_yinList = src_yin.split(' ')
            gt_yinList = gt_yin.split(' ')
            for i, yin in enumerate(src_yinList):
                sum_num += 1
                if yin == gt_yinList[i]:
                    acc_num += 1

print("行准确率:{}".format(acc_line/sum_line))
print("字准确率:{}".format(acc_num/sum_num))

with open(out_path,'w') as out_file:
    out_file.write("行准确率:{}".format(acc_line/sum_line)+'\n')
    out_file.write("字准确率:{}".format(acc_num/sum_num))

