from pyvi import ViTokenizer
from utils import filter_punctuation
from sklearn.model_selection import train_test_split
from rule_noise import *
import copy
import json
import random
import re
import numpy as np
import string 

percentage_of_sentence = 30 # số lượng noise dao động trong một câu, từ 0 đến 30% theo chiều dài của câu 
number_of_data = 0.7 # số lượng dữ liệu làm nhiều -  tính theo phần trăm  
# data_original => get_original_data
def get_original_data(file, path):
    """
        ham nay chi doc du lieu anh minh gui
        data1 - len raw va original = nhau
        data2 - len raw va original khac nhau
    """
    data1 = []
    data2 = []
    with open(file, 'r') as json_data:
        for f in json_data:
            tmp = json.loads(f)  
            x = copy.copy(tmp['raw'])
            y = ViTokenizer.tokenize(x)
            y = filter_punctuation(y)  # loc dau cau
            y = y.split(" ")
            tmp['raw'] = copy.copy(y)
            x1 = copy.copy(tmp['original'])
            y1 = ViTokenizer.tokenize(x1)
            y1 = filter_punctuation(y1)  # loc dau cau
            y1 = y1.split(" ")
            tmp['original'] = copy.copy(y1)
            tmp.update({'tid': 0})
            if len(y) == len(y1):
                data1.append(tmp)
            else:
                data2.append(tmp)
    with open(path+'data_original.json', 'w') as outfile:
        json.dump(data1, outfile, ensure_ascii=False)
        
    print(len(data1))

    print(len(data1) + len(data2))

def convert(fil, data_label):
    """
        chọn riêng các item cùng loại ra 
    """
    data = []
    data_element = []
    with open(fil, 'r') as json_data:
        for element in json_data:
            data.append(json.loads(element))

    for element in data:
        tmp = element['id'].split("_")
        if(tmp[0] == data_label):
            data_element.append(element)
    return data_element

def split_token_json(data):
    """  
        Từ dữ liệu ban đầu, tiến hành đọc dữ liệu và tokienize thành các từ
        rồi lọc dấu câu 

        Chuyển câu thành list các từ 

        'original' - là dữ liệu đúng
        'raw' - là dữ liệu sai sau khi tạo nhiều từ dữ liệu đúng  

        bổ sung thêm key 'tid' theo format dữ liệu trong repos cũ 

    """         
    for i in range(len(data)):
        x1 = copy.copy(data[i]['original'])
        y1 = ViTokenizer.tokenize(x1)
        y1 = filter_punctuation(y1)  # loc dau cau
        y1 = y1.split(" ")
        data[i]['original'] = copy.copy(y1)
        data[i]['raw'] = copy.copy(y1)
        data[i].update({'tid': 0})
    return data
    
def select_word(tmp, mark_sequen):
    """
        n là vị trị của từ
        mark_sequen đánh dấu vị trí các từ đã được chọn để  làm nhiễu
    """
    n = random.randint(0, len(tmp['original']) - 1)
    if mark_sequen[n] == 1:
        n = random.randint(0, len(tmp['original']) - 1)
        while mark_sequen[n] == 1:
            n = random.randint(0, len(tmp['original']) - 1)
        mark_sequen[n] = 1
        word = tmp['original'][n]
    else:
        mark_sequen[n] = 1
        word = tmp['original'][n]
    return n, word, mark_sequen

def add_noise_sequen(data1, f):
    """
        lấy 70 phần trăm dữ liệu để tạo thêm 
        môi câu được chọn sẽ chọn số lượng lỗi bằng 30% chiều dài câu (một từ không được chọn lại) -- chính là số từ bị lỗi 
        vẫn giữ cả câu ban đầu 
        data1 - dữ liệu ban đầu
        data2 - dữ liệu tạo ra 
        fff-file - lưu các từ được tạo ra file 
        sau đó trộn data2 vào data1 
    """
    
    print('len data ban dau {}'.format(len(data1)))
    data2 = []  # noise tao ra
    regex1 = re.compile(r"\S*\d+\S*", re.UNICODE) # lọc số 
    regex2 = re.compile(
        r'^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz_]+$', re.UNICODE) 
    error = np.arange(26)
    sequence = [1, 2, 3]
    # dem so loi
    file_errorr = []
    for i in range(26):
        file_errorr.append(0)
    random.seed(3255)
    element_random = random.sample(range(len(data1)), int(number_of_data*len(data1)))
    for i in range(len(data1)):
        i1 = i
        if i1 in element_random:  # xác suất chọn câu để thêm nhiễu 
            n_quence = random.choice(sequence)
            mark_sequen = [0]*len(data1[i]['original'])
            for j in range(n_quence - 1):
                n_error = (random.randint(0, percentage_of_sentence)*len(data1[i]['original']))/100 # nếu câu quá ngắn thì sẽ không tạo ra câu nào, do giá trị n_erorr = 0
                tmp = copy.deepcopy(data1[i])
                tmp['id'] = tmp['id'] + str(j)
                # do câu quá ngắn nên sẽ sét mặc đinh là có một lỗi 
                if int(n_error) == 0: 
                    n_error = 1
                # nếu câu có 1 từ mà cần tạo thêm 2 câu thì khi chạy sẽ bị lỗi do không tim được từ nào khác từ trước 
                if len(tmp['original']) > 1: 
                    for j1 in range(int(n_error)):
                        # chọn ngẫu nhiên vị trí một từ 
                        n, word, mark_sequen = select_word(tmp, mark_sequen)
                        # số thì loại, chỉ các từ chứa các chữ cái trong bẳng chữ cái, các dấu câu không tính
                        if not re.search(regex1, word) and re.search(regex2, word) :
                            op = random.choice(error)
                            file_word1 = copy.copy(word)
                            word = add_noise(word, op)
                            file_word2 = copy.copy(word)
                            # nếu sau khi tạo noise mà không tạo ra được từ mới thì chon loại noise khác 
                            while file_word1 == file_word2:
                                op = random.choice(error)
                                word = add_noise(word, op)
                                file_word2 = copy.copy(word)
                            file_errorr[op] = file_errorr[op] + 1
                            f.write('%-15s  <%-2d>  %-15s\n' %
                                        (file_word1, op, file_word2))
                            tmp['raw'][n] = word
                            data2.append(tmp)
    # thong ke cac loi khi them                
    print(file_errorr)  
    print('length data noise tao ra {}'.format(len(data2)))
    data1 = data1 + data2 
    print('sum data {}'.format(len(data1)))
    return data1

def add_noise(word, op):
    """
    ------ 0. 13  xóa 1 kí tự trong từ, ngoại trừ underscore

    ------- 1, 14 đổi chỗ 2 kí tự liên tiếp trong một từ
       90% với các trường hợp từ có ch, tr, kh, nh, ng,, qu, th, ph, gh, gi, ngh
       10% với các trường hợp còn lại 
    --------2. 15  thêm underscore trong một từ, không thêm vào cuối của từ 
    --------3. 16  bỏ ngẫu nhiên một underscore trong một từ nếu có 
    --------4. thay một kí tự bằng môt kí tự gần nhau trên bàn phím 
    --------5. dùng noise telex 
        có 3 kiểu:
            kiểu một: thay tại chỗ
                vd: mẩy ====> maary (1), maray (2)
            kiểu hai: thêm dấu ở cuối của từ
                vd: mẩy ====> maayr (3)
            kiểu ba: viết tất cả ở cuối của từ
                vd mẩy =====> mayar (4), mayra (5)
            90% cho trường hợp (1) và (3)
            10% với các trường hợp còn lại 
          
    --------6. tương tự noise vni
    --------7. lập lại một số lần từ 1 - 5 các nguyên âm ouieay mà nó không dứng ở đầu 
    --------8. cac am vi o dau cua am tiet gan giong nhau
    --------8. 17. 18 các âm vị gần giống nhau ở đầu của âm tiết 
    --------9. 19.20 các âm vị ở cuối gần giống nhau của âm tiết, theo saigon 
    --------10 , 21  thay đổi dấu ngã và hỏi đối với các nguyên âm 
    --------11 , 22, 23 các âm vị ở đầu có cách phát âm giống nhau trong một số trường hợp ví dụ: c,q,k
    --------12, 24, 25 sai vi tri dau va thay doi chu cai voi to hop cac dau neu am tiet co 1 nguyen am
    --------12, 24, 25 sai vị trí dấu
        có 2 trường hợp :  <> nếu âm tiết chỉ có một nguyên âm, thì sẽ đối ngẫu nhiên các trường hợp được ghi trong
        get_change_sign
            vd: a <==> ngẫu nhiên: à', 'á', 'â', 'ã', 'ạ','ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ'
                           <> nếu có âm đêm và âm chính hoặc có thêm âm cuối thì chuyển dấu sang âm đệm
            vd: òa oà
    """

    i = random.randint(0, len(word) - 1)
    # op = random.randint(0,12)
    # print(op)
    # i  = 1
    if op == 0 or op == 13:
        if word[i] != '_' and len(word) >= 2:
            return word[:i] + word[i+1:]
    if op == 1 or op == 14:
        i += 1
        consonants = ['ch', 'tr', 'kh', 'nh', 'ng',
                      'qu', 'th', 'ph', 'gh', 'gi', 'ngh']  # 11
        tmp = [i for i in consonants if i in word]
        r = random.random()
        if len(tmp) > 0 and r > 0.1:
            tmp1 = random.choice(tmp)
            if tmp1 == 'ngh':
                return word.replace(tmp1, random.choice(consonant_trigraphs(tmp1)))
            else:
                return word.replace(tmp1, consonant_digraphs(tmp1))
        elif i <= len(word) - 1:
            return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]
        else:
            return word
    if op == 2 or op == 15:
        i += 1
        if i <= len(word) - 1:
            return word[:i] + '_' + word[i:]
        else:
            return word
    if op == 3 or op == 16:
        idx = word.find("_")  # check xem co underscore khong
        if idx != -1:
            list_underscore = [m.start() for m in re.finditer('_', word)]
            idx1 = random.choice(list_underscore)
            return word[:idx1] + word[idx1 + 1:]
        else:
            return word
    if op == 4:
        string_list = string.ascii_lowercase
        if word[i] in string_list:
            return word[:i] + random.choice(get_prox_keys(word[i])) + word[i+1:]
        else:
            return word
    # tam thoi coi 2 cai duoi la 1
    if op == 5:
        string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
        syllable = word.split('_')
        tmp = []
        for syllable_i in syllable:
            letter = [str(i) for i in string_list if i in syllable_i]
            for letter_i in letter:
                tmp1 = noise_telex(letter_i)
                # print(tmp1)
                r = random.random()
                if r < 0.9:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'str':
                            # print('1')
                            syllable_i = syllable_i.replace(letter_i, tmp1[0])

                        else:
                            # print('2')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][0])
                            # return syllable_i

                    else:
                        if type(tmp1[1]).__name__ == 'str':
                            # print('3')
                            syllable_i = syllable_i.replace(letter_i, tmp1[1])

                        else:
                            # print('4')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[1][0]) + tmp1[1][1]

                            # return syllable_i
                else:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'list':
                            # print('5')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][1])

                    else:
                        # print('6')
                        syllable_i = syllable_i.replace(
                            letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
            tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word

    if op == 6:
        string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
        syllable = word.split('_')
        tmp = []
        for syllable_i in syllable:
            letter = [str(i) for i in string_list if i in syllable_i]
            for letter_i in letter:
                tmp1 = noise_vni(letter_i)
                # print(tmp1)
                r = random.random()
                if r < 0.9:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'str':
                            # print('1')
                            syllable_i = syllable_i.replace(letter_i, tmp1[0])

                        else:
                            # print('2')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][0])
                            # return syllable_i

                    else:
                        if type(tmp1[1]).__name__ == 'str':
                            # print('3')
                            syllable_i = syllable_i.replace(letter_i, tmp1[1])

                        else:
                            # print('4')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[1][0]) + tmp1[1][1]

                            # return syllable_i
                else:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'list':
                            # print('5')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][1])

                    else:
                        # print('6')
                        syllable_i = syllable_i.replace(
                            letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
            tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word
    if op == 7:
        l = word[i]
        vowel = 'ouieay'
        if i >= 1 and l in vowel:
            return word[:i] + random.randint(1, 5) * l + word[i+1:]
        else:
            return word

    # thay doi o dau
    if op == 8 or op == 17 or op == 18:
        string_list1 = ['l', 'n', 'x', 's', 'r', 'd', 'v']
        string_list2 = ["ch", 'tr', 'gi']
        syllable = word.split("_")
        for i, syllable_i in enumerate(syllable):
            r = random.random()
            if r > 0.5:
                if len(syllable_i) >= 1 and syllable_i[0] in string_list1:
                    wo = random.choice(closely_pronunciation1(syllable_i[0]))
                    syllable[i] = wo + syllable_i[1:]
                elif len(syllable_i) >= 2 and syllable_i[0]+syllable_i[1] in string_list2:
                    wo = random.choice(closely_pronunciation1(
                        syllable_i[0] + syllable_i[1]))
                    syllable[i] = wo + syllable_i[2:]
        return "_".join(syllable)
    # thay doi o cuoi
    # co mot vai truong hop rieng thoi
    # saigon phonology
    if op == 9 or op == 19 or op == 20:
        string_list1 = ['inh', 'ênh', 'iên', 'ươn', 'uôn', 'iêt', 'ươt', 'uôt']
        string_list2 = ['ăn', 'an', 'ân', 'ưn', 'ắt', 'ât', 'ưt', 'ôn', 'un',
                        'ât', 'ưt', 'ôn', 'un', 'ôt', 'ut']
        syllable = word.split("_")
        tmp = []
        for syllable_i in syllable:
            if len(syllable_i) >= 3 and syllable_i[len(syllable_i) - 3:] in string_list1:
                syllable_i = syllable_i[:len(
                    syllable_i) - 3] + saigon_final3(str(syllable_i[len(syllable_i) - 3:]))
                tmp.append(syllable_i)
            elif len(syllable_i) >= 2 and syllable_i[len(syllable_i) - 2:] in string_list2:
                syllable_i = syllable_i[:len(
                    syllable_i) - 2] + saigon_final2(str(syllable_i[len(syllable_i) - 2:]))
                tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word

    if op == 10 or op == 21:
        string_list = ['ã', 'ả',
                       'ẫ', 'ẩ',
                       'ẵ', 'ẳ',
                       'ẻ', 'ẽ',
                       'ể', 'ễ',
                       'ĩ', 'ỉ',
                       'ũ', 'ủ',
                       'ữ', 'ử',
                       'õ', 'ỏ',
                       'ỗ', 'ổ', 'ỡ', 'ở']
        swap = {'ã': 'ả', 'ả': 'ã', 'ẫ': 'ẩ', 'ẩ': 'ẫ',
                'ẵ': 'ẳ', 'ẳ': 'ẵ', 'ẻ': 'ẽ', 'ẽ': 'ẻ', 'ễ': 'ể', 'ể': 'ễ',
                'ĩ': 'ỉ', 'ỉ': 'ĩ', 'ũ': 'ủ', 'ủ': 'ũ', 'ữ': 'ử', 'ử': 'ữ',
                'õ': 'ỏ', 'ỏ': 'õ', 'ỗ': 'ổ', 'ổ': 'ỗ', 'ỡ': 'ở', 'ở': 'ỡ'}
        tmp = [i for i in string_list if i in word]
        for letters in tmp:
            word = word.replace(letters, swap[letters])
        return word

    if op == 11 or op == 22 or op == 23:
        string_list0 = ['ngh']
        string_list1 = ['gh', 'ng']
        string_list2 = ['g', 'c', 'q', 'k']
        syllable = word.split("_")
        for i, syllable_i in enumerate(syllable):
            r = random.random()
            if r > 0.5:
                if len(syllable_i) >= 3 and syllable_i[0] + syllable_i[1] + syllable_i[2] in string_list0:
                    wo = random.choice(like_pronunciation2(
                        syllable_i[0] + syllable_i[1] + syllable_i[2]))
                    syllable[i] = wo + syllable_i[3:]
                elif len(syllable_i) >= 2 and syllable_i[0] + syllable_i[1] in string_list1:
                    wo = random.choice(like_pronunciation2(
                        syllable_i[0] + syllable_i[1]))
                    syllable[i] = wo + syllable_i[2:]
                elif len(syllable_i) >= 1 in string_list2:
                    wo = random.choice(like_pronunciation2(syllable_i[0]))
                    syllable[i] = wo + syllable_i[1:]
        return "_".join(syllable)

    """
        thay doi vi tri dau
    """

    string_list1 = 'àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵaeiouy'
    string_list2 = ['óa', 'oá', 'òa','oà', 'ỏa', 'oả', 'õa', 'oã', 'ọa', 'oạ',\
            'áo', 'aó', 'ào','aò', 'ảo', 'aỏ', 'ão', 'aõ', 'ạo', 'aọ',\
            'éo', 'eó', 'èo','eò', 'ẻo', 'eỏ', 'ẽo', 'eõ', 'ẹo', 'eọ',\
            'óe', 'oé', 'òe','oè', 'ỏe', 'oẻ', 'õe', 'oẽ', 'ọe', 'oẹ',\
            'ái', 'aí', 'ài','aì', 'ải', 'aỉ', 'ãi', 'aĩ', 'ại', 'aị',\
            'ói', 'oí', 'òi','oì', 'ỏi', 'oỉ', 'õi', 'oĩ', 'ọi', 'oị'] # convert ve khong dau 

    dict_change = {'óa': 'oá', 'òa':'oà', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',\
                    'oá': 'óa', 'oà':'òa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',\
                    'áo': 'aó', 'ào':'aò', 'ảo': 'aỏ', 'ão': 'aõ', 'ạo': 'aọ',\
                    'aó': 'áo', 'aò':'ào', 'aỏ': 'ảo', 'aõ': 'ão', 'aọ': 'ạo',\
                    'éo': 'eó', 'èo':'eò', 'ẻo': 'eỏ', 'ẽo': 'eõ', 'ẹo': 'eọ',\
                    'eó': 'éo', 'eò':'èo', 'eỏ': 'ẻo', 'eõ': 'ẽo', 'eọ': 'ẹo',\
                    'óe': 'oé', 'òe':'oè', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',\
                    'oé': 'óe', 'oè':'òe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe', 'ái': 'aí', 'ài':'aì', 'ải': 'aỉ', 'ãi': 'aĩ', 'ại': 'aị', 'aí': 'ái', 'aì':'ài', 'aỉ': 'ải', 'aĩ': 'ãi', 'aị': 'ại', 'ói': 'oí', 'òi':'oì', 'ỏi': 'oỉ', 'õi': 'oĩ', 'ọi': 'oị',\
                    'oí': 'ói', 'oì':'òi', 'oỉ': 'ỏi', 'oĩ': 'õi', 'oị': 'ọi'}

    syllable = word.split("_")
    word_add = []
    for i, syllable_i in enumerate(syllable):
        tmp = [i for i in string_list1 if i in syllable_i]
        if len(tmp) == 1:
            syllable_i = syllable_i.replace(tmp[0], random.choice(get_change_sign(tmp[0])))
            word_add.append(syllable_i)
        else :
            tmp1 = [i for i in string_list2 if i in syllable_i]
            for letters in tmp1:
                syllable_i = syllable_i.replace(letters, dict_change[letters])
                word_add.append(syllable_i)
    if len(word_add) != 0:
        return "_".join(word_add)
    else:
        return word

def read_file_ducanh(file, label):
    data = []
    with open(file, 'r') as json_data:
            tmp = json_data.readlines()
            count = 0
            for line in tmp:
                json_data = {}
                x1 = copy.copy(line)
                y1 = ViTokenizer.tokenize(x1)
                y1 = filter_punctuation(y1)
                y1 = y1.split(" ")
                json_data['original'] = copy.copy(y1)
                json_data['raw'] = copy.copy(y1)
                json_data.update({'tid' : 0})
                json_data.update({'id' : label+ str(count)})
                data.append(json_data)
                count +=1
    return data 

def read_data_ducanh(file_ducanh):
    """
        thêm dữ liệu của dức anh từ 2 file 
    """

    data = []
    data1 =  read_file_ducanh(file_ducanh[0], 'DUCANH1')
    data2 =  read_file_ducanh(file_ducanh[1], 'DUCANH2')
    data = data1 + data2
    return data

def merge_data_noise(data, data_ducanh, path,final_json):
    """
        data được chia theo tỉ lệ 8:2 
        và ghi ra file 'train_data.json', 'test_data.json' để dùng cho model 
    """
    total_train, total_test = train_test_split(data, test_size=0.2, random_state=3255)
    print('tong tat ca du lieu anh minh  {}'.format(len(total_train)+len(total_test)))
    train, test = train_test_split(data_ducanh, test_size=0.2, random_state=3255)
    total_train = total_train + train
    total_test = total_test + test
    print('tong tat ca du lieu cua anh minh va duc anh  {}'.format(len(total_train)+len(total_test)))
    # ghi vao file train va test o dang list de dua vao model
    with open(path+final_json[0], 'w') as outfile:
        json.dump(total_train, outfile, ensure_ascii=False)
    with open(path+final_json[1], 'w') as outfile:
        json.dump(total_test, outfile, ensure_ascii=False)
