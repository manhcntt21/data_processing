#!/usr/bin/python

# -*- coding: utf8 -*-
from sklearn.utils import shuffle
from preprocessing import get_original_data, convert, split_token_json, add_noise_sequen, merge_data_noise, read_data_ducanh
from utils import filter_punctuation, test_length
import random
random.seed(3255)
path = './data111/'
file = 'data_.jsonl'
data_labels = ["BOOK", "CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
final_json = ['train_data.json', 'test_data.json']
file_ducanh = ['./data_cuong/2016/train2016-refine.txt',
               './data_cuong/2018/train2018-refine.txt']

# xử lý riêng với dữ liệu của đức anh, không cần phải lọc theo item vì có một loại
def custom_data(f):
    tokenize_data = read_data_ducanh(file_ducanh)
    noisy_data = add_noise_sequen(tokenize_data, f)  # add_noise
    data = shuffle(noisy_data)  # shuffle data
    return data

if __name__ == '__main__':
    f = open("text.txt", "w+")
    data = []
    for i in range(len(data_labels)):
        raw_data = convert(file, data_labels[i])
        tokenize_data = split_token_json(raw_data)
        noisy_data = add_noise_sequen(tokenize_data, f)  # add_noise
        data_element = shuffle(noisy_data)  # shuffle data
        data += data_element
    data_ducanh = custom_data(f)
    # trộn 2 loại data và ghi vào file: final_json 
    merge_data_noise(data, data_ducanh, path, final_json)
