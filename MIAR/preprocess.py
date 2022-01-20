# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_preprocess
   Description :
   Author :       killjl
   date：          2021/11/25
-------------------------------------------------
"""
import json
import random
import csv
import re

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3
AUG = 2

# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

def data_split(source, fold_num):

    all = []
    InfoG = []
    InfoS = []
    ProD = []
    Other = []

    with open(source, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmt = row['review']
            if len(cmt) > 50:
                continue
            lable = row['intention']
            all.append(row)
            if lable == 'INFORMATION SEEKING':
                InfoS.append(cmt)
                InfoS.append(lable)
                for i in range(AUG):
                    sentence_aug = insert_punctuation_marks(cmt)
                    InfoS.append(sentence_aug)
                    InfoS.append(lable)
            elif lable == 'INFORMATION GIVING':
                InfoG.append(cmt)
                InfoG.append(lable)
            elif lable == 'PROBLEM DISCOVERY':
                ProD.append(cmt)
                ProD.append(lable)
            else:
                Other.append(cmt)
                Other.append(lable)
    
    print(len(all))
    class_statistic(InfoG , InfoS , ProD , Other)
    lenOther = 120000
    InfoG_fold_num = len(InfoG) // fold_num
    InfoS_fold_num = len(InfoS) // fold_num
    ProD_fold_num = len(ProD) // fold_num
    UE_fold_num = lenOther // fold_num
    f_folds = []
    b_folds = []
    r_folds = []
    u_folds = []
    for i in range(fold_num):
        if i == fold_num - 1:
            f_folds.append(InfoG[i * InfoG_fold_num:(i + 1) * InfoG_fold_num]) #i * InfoG_fold_num
            b_folds.append(InfoS[i * InfoS_fold_num:(i + 1) * InfoS_fold_num]) #i * InfoS_fold_num
            r_folds.append(ProD[i * ProD_fold_num:(i + 1) * ProD_fold_num]) #i * ProD_fold_num
            u_folds.append(Other[i * UE_fold_num:(i + 1) * UE_fold_num]) #i * UE_fold_num
        else:
            f_folds.append(InfoG[i * InfoG_fold_num:(i + 1) * InfoG_fold_num])
            b_folds.append(InfoS[i * InfoS_fold_num:(i + 1) * InfoS_fold_num])
            r_folds.append(ProD[i * ProD_fold_num:(i + 1) * ProD_fold_num])
            u_folds.append(Other[i * UE_fold_num:(i + 1) * UE_fold_num])
    train_folds = []
    test_folds = []
    for i in range(fold_num):
        train = []
        test = []
        for j in range(fold_num):
            if j == i:
                test.extend(f_folds[j])
                test.extend(b_folds[j])
                test.extend(r_folds[j])
                test.extend(u_folds[j])
            else:
                train.extend(f_folds[j])
                train.extend(b_folds[j])
                train.extend(r_folds[j])
                train.extend(u_folds[j])
        train_folds.append(train)
        test_folds.append(test)

    return train_folds, test_folds


def class_statistic(InfoG , InfoS , ProD , Other):
    InfoG_cnt = len(InfoG)/2
    InfoS_cnt = len(InfoS)/2
    ProD_cnt = len(ProD)/2
    UE_cnt = len(Other)/2

    print(f"Infomation Giving: {InfoG_cnt}, Infomation Seeking report: {InfoS_cnt}, Problem Discovery: {ProD_cnt}, Other: {UE_cnt}, All: {InfoG_cnt + InfoS_cnt + ProD_cnt + UE_cnt }")


def make_data():
    fold_num = 3
    train_folds, test_folds = data_split(f"data/review-data.csv", fold_num)
    for i in range(fold_num):
        train = train_folds[i]
        test = test_folds[i]
        with open(f"data/train_{i}.txt", "a") as f:
            for line in train:
                f.write(line + '\n')
        
        with open(f"data/test_{i}.txt", "a") as f:
            for line in test:
                f.write(line + '\n')


if __name__ == "__main__":
    make_data()
