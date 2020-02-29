#!/usr/bin/python

# -*- coding: utf8 -*-
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocessing import convert, split_token_json, add_noise_sequen, merge_data_noise, read_data_ducanh
from preprocessing import get_length_data_json, get_length_data_add
import random
import os
import shutil

random.seed(3255)
file = 'data_.jsonl'
data_labels = ["BOOK", "CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
final_json = ['train_data.json', 'test_data.json']
file_ducanh = ['./data_cuong/2016/train2016-refine.txt',
               './data_cuong/2018/train2018-refine.txt']

# tạo dữ liệu 
# path = './data_8_2/'
# rate_train_test = 0.2 # train:test=8:2

path = './data_9_1/'
rate_train_test = 0.1 # train:test=9:1

# xử lý riêng với dữ liệu của đức anh, không cần phải lọc theo item vì có một loại
def custom_data(f):
    print('- Dữ liệu bổ sung thêm')
    tokenize_data = read_data_ducanh(file_ducanh)
    train, test = train_test_split(tokenize_data, test_size=rate_train_test, random_state=3255)
    print("\t- -----------------TRAIN-----------------------")
    noisy_data_train = add_noise_sequen(train, f)  # add_noise
    print("\t- -----------------TEST-----------------------")
    noisy_data_test = add_noise_sequen(test, f)
    data_train = shuffle(noisy_data_train)  # shuffle data
    data_test = shuffle(noisy_data_test)  # shuffle data
    return data_train, data_test

def precessing_element(element_data, f):
    """
        Chyển raw data thành format dữ liệu theo repos (có thêm tid) và chuyển câu thành các từ
        Thêm noise 
        hoán vị dữ liệu 
    """
    tokenize_data = split_token_json(element_data)
    noisy_data = add_noise_sequen(tokenize_data, f)
    data = shuffle(noisy_data)

    return data
if __name__ == '__main__':
    f = open("text.txt", "w+")
    data = []
    train = []
    test = []
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)
    print('# '+(path))
    length_a = get_length_data_json(file)
    length_b = get_length_data_add(file_ducanh)
    print('* **Kích thước dữ liệu gốc của anh Minh {}**'.format(length_a))
    print('* **Kích thước dữ liệu gốc của Đức Anh {}**'.format(length_b))
    for i in range(len(data_labels)):
        print('- Dữ liệu {}'.format(data_labels[i]))
        raw_data = convert(file, data_labels[i])
        element_train, element_test = train_test_split(raw_data, test_size=rate_train_test, random_state=3255)
        print("\t- -----------------TRAIN-----------------------")
        element_train = precessing_element(element_train, f)
        print("\t- -----------------TEST-----------------------")
        element_test = precessing_element(element_test, f)
        train += element_train
        test += element_test
    data_ducanh_train, data_ducanh_test = custom_data(f)
    
    train += data_ducanh_train
    test += data_ducanh_test
    # trộn 2 loại data và ghi vào file: final_json 
    merge_data_noise(train, test, path, final_json)
    f.close()
