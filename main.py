#!/usr/bin/python

# -*- coding: utf8 -*-

import json

from pyvi import ViTokenizer, ViPosTagger
import random
import re
import copy
from sklearn.utils import shuffle
import string
import collections
import csv
from unidecode import unidecode 
import numpy as np
from preprocessing import get_original_data, convert, split_train_test, split_token_json, add_noise_sequen, merge_data_noise, read_data_ducanh
from utils import filter_punctuation, filter_number, test_length


random.seed(3255)

# sinh ra data loai 1
path = './data111/'
file = 'data_.jsonl'
data_label = ["BOOK", "CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
final_json = ['train_data.json', 'test_data.json']
file_ducanh = ['./data_cuong/2016/train2016-refine.txt', './data_cuong/2018/train2018-refine.txt']
def data_add_noise(data_element, fff_file):
    data_element = split_token_json(data_element) 
    data_element = add_noise_sequen(data_element, fff_file)  # add_noise
    data_element = shuffle(data_element)  # shuffle data
    return data_element
def result():
    fff_file = open("text.txt", "w+")
    data = []
    for i in range(len(data_label)):
        data_element = convert(file, data_label[i])
        data_element = data_add_noise(data_element, fff_file)
        data += data_element
    # xử lý riêng với dữ liệu của đức anh, không cần phải lọc theo item vì có một loại 
    data_ducanh = []    
    data_ducanh = read_data_ducanh(file_ducanh) 
    data_ducanh = add_noise_sequen(data_ducanh, fff_file)  # add_noise
    data_ducanh = shuffle(data_ducanh)  # shuffle data
    merge_data_noise(data, data_ducanh, path, final_json)


if __name__ == '__main__':
    result()
    # test_length(path,'train_data.json')
    # test_length(path,'test_data.json')
    # test_length(path,'test_data.json', 'test_data.json')
    # create_tiny()

