import re
import json


def filter_punctuation(my_str):
    """
        chi nhung tu nay moi can loc
        regex2  = re.compile(
            '^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$',re.UNICODE)
    j = 0
    for i in b:
                if re.search(regex2,i):
                # print(add_noise(i,0))
                j+=1
                sau khi tokenzier thi loc

    """
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
        else:
            no_punct = no_punct + ' @' + char
    return no_punct

def filter_number(data):
    """
        thêm @ vào các phần tử chứa số trong raw và original
        nếu một từ có cả dấu câu và số thì sẽ có 2 chữ @ 
    """
    regex1 = re.compile(r"\S*\d+\S*", re.UNICODE)
    for i in range(len(data)):
        for j, k in enumerate(data[i]['original']):
            if re.search(regex1, k):
                data[i]['raw'][j] = '@' + data[i]['raw'][j]
                data[i]['original'][j] = '@' + data[i]['original'][j]

    return data

def test_length(path, f1):
    # test length cau
    with open(path+f1, 'r') as json_data:
        data = json.load(json_data)
    max_ = 0
    for i in data:
        if max_ < len(i['original']):
            max_ = len(i['original'])
    print(max_)
    print('len cau', len(data))
    # 68 + 2 = 70
    # 65 + 2 = 67 =>>>>>>>69
    # test length tu

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
    # 47 + 4 =
    # 23 + 4 = 25  ========>
