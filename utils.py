import re
import json

def filter_punctuation(my_str):
    """
        thêm @ vào các dấu câu 
    """
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
        else:
            no_punct = no_punct + ' @' + char
    return no_punct

def test_length(path, f1):
    """
        tính chiều dài của câu và chiều dài của từ , từ file

        cần phải công thêm 2 vì khi áp dung vào mô hình cần bổ sung thêm 2 kí tự bắt đầu và kết thúc 
    """
    with open(path+f1, 'r') as json_data:
        data = json.load(json_data)
    max_ = 0
    for i in data:
        if max_ < len(i['original']):
            max_ = len(i['original'])
    print(max_)
    print('len cau', len(data))

    max_ = 0
    for i in data:
        tmp1 = i['raw']
        tmp2 = i['original']
        for j in tmp1:
            if max_ < len(j):
                max_ = len(j)
                # print(i['id'])
        for k in tmp2:
            if max_ < len(k):
                max_ = len(k)
                # print(i['id'])
    print(max_)
    print('len word ', len(data))



if __name__ == '__main__':
    test_length('./data_8_2/','train_data.json')
    test_length('./data_8_2/','test_data.json')
    # get_length_data_json('data_.jsonl')
    # file_ducanh = ['./data_cuong/2016/train2016-refine.txt',
    #            './data_cuong/2018/train2018-refine.txt']
    # get_length_data_add(file_ducanh)
    pass
