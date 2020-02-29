from pyvi import ViTokenizer
from utils import filter_punctuation
from rule_noise import *
from sklearn.utils import shuffle
import copy
import json
import random
import re
import numpy as np
import string 


percentage_of_sentence = 30 # số lượng noise dao động trong một câu, từ 0 đến 30% theo chiều dài của câu 
number_of_data = 0.7 # số lượng dữ liệu làm nhiều -  tính theo phần trăm  
# data_original => get_original_data
def get_data_book(file, path = None):
    """

    """
    data = []
    with open(file, 'r') as json_data:
        for f in json_data:
            tmp = json.loads(f)  
            x1 = copy.copy(tmp['original'])
            y1 = ViTokenizer.tokenize(x1)
            # y1 = filter_punctuation(y1)  # loc dau cau
            y1 = y1.split(" ")
            tmp['original'] = copy.copy(y1)
            tmp['raw'] = copy.copy(y1)
            tmp.update({'tid': 0})
            data.append(tmp)
    # with open(path+'data_original.json', 'w') as outfile:
    #     json.dump(data1, outfile, ensure_ascii=False)
    return data    

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
        # y1 = filter_punctuation(y1)  # loc dau cau
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
        f - lưu các từ được tạo ra file 
        sau đó trộn data2 vào data1 
    """
    
    print('\t\t- lượng dữ liệu ban đầu {}'.format(len(data1)))
    data2 = []  # noise tao ra
    regex1 = re.compile(r"\S*\d+\S*", re.UNICODE) # lọc số 
    regex2 = re.compile(
        r'^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz_]+$', re.UNICODE) 
    error = np.arange(26)
    sequence = [1, 2, 3]
    # dem so loi
    element_erorr = []
    for i in range(26):
        element_erorr.append(0)
    # gộp các loại lỗi lại
    final_erorr = []
    for i in range(13):
        final_erorr.append(0)
    random.seed(3255)
    element_random = random.sample(range(len(data1)), int(number_of_data*len(data1)))
    for i in range(len(data1)):
        i1 = i
        if i1 in element_random:  # xác suất chọn câu để thêm nhiễu 
            n_quence = random.choice(sequence) # sinh thêm 1 hoặc 2 hoặc 3 câu từ câu đã chọn
            mark_sequen = [0]*len(data1[i]['original'])
            for j in range(n_quence): # id của các câu tạo thêm sẽ được bổ sung thêm 0 1 2
                n_error = (random.randint(0, percentage_of_sentence)*len(data1[i]['original']))/100 # nếu câu quá ngắn thì sẽ không tạo ra câu nào, do giá trị n_erorr = 0
                tmp = {}
                tmp = copy.deepcopy(data1[i])
                # do câu quá ngắn nên sẽ sét mặc đinh là có một lỗi 
                if int(n_error) == 0: 
                    n_error = 1
                # nếu câu có 1 từ mà cần tạo thêm 2 câu thì khi chạy sẽ bị lỗi do không tim được từ nào khác từ trước 
                
                # câu quá ngắn có từ kèm theo 1 dấu câu, khi tách từ thì cậu có độ dài bằng 2, thì chỉ có thể tạo thêm được 1 câu thôi.
                # vì từ kia đã được làm lỗi rồi 
                # chưa tìm ra cách đểm các phần tử thật sư là từ trong một câu, vì một từ có dấu _ ở giữa các âm tiết, mà trong khi tách từ cũng có thể  tạo 
                # các phần tử là dấu cách, tạm thời dùng điều kiện ở dưới 
                if len(tmp['original']) > 2: 
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
                            element_erorr[op] = element_erorr[op] + 1
                            try:
                                f.write('%-15s  <%-2d>  %-15s\n' %(file_word1, op, file_word2))
                            except:
                                pass                 
                            tmp['raw'][n] = word

                # nếu sau khi thử hết các trường hợp mà câu đó không tạo thêm từ mới thì bỏ 
                # chẳng hạn nếu câu có 1 từ nhưng lại được tạo thêm 2 câu thì không thể, vì một từ chỉ có một lỗi  
                # elif sum(mark_sequen) == len(tmp['original']):
                #     break

                if tmp != data1[i1]:
                    tmp['id'] = tmp['id'] + 'MANH' + str(j) 
                    data2.append(tmp)

    # thong ke cac loi khi them                
    final_erorr = get_statistical_erorr(final_erorr, element_erorr)
    print('\t\t- thông kê các loại lỗi trong dữ liệu')
    print('\t\t- {}'.format(final_erorr))
    print('\t\t- tổng dữ liệu noise tạo ra {}'.format(len(data2)))
    data1 = data1 + data2 
    print('\t\t- tổng dữ liệu {}'.format(len(data1)))
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
    --------8. 17. 18 các âm vị gần giống nhau ở đầu của âm tiết 
    --------9. 19.20 các âm vị ở cuối gần giống nhau của âm tiết, theo saigon 
    --------10 , 21  thay đổi dấu ngã và hỏi đối với các nguyên âm 
    --------11 , 22, 23 các âm vị ở đầu có cách phát âm giống nhau trong một số trường hợp ví dụ: c,q,k
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
                # y1 = filter_punctuation(y1)
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
        đọc dữ liệu của dức anh từ 2 file 
    """

    data = []
    data1 =  read_file_ducanh(file_ducanh[0], 'DUCANH_')
    data2 =  read_file_ducanh(file_ducanh[1], 'DUCANH__')
    data = data1 + data2
    return data

def merge_data_noise(train, test, path,final_json):
    """
        chỉ việc ghi dữ liệu train, test ra file 
    """
    # total_train, total_test = train_test_split(data, test_size=0.2, random_state=3255)
    # print('tổng dữ liệu của anh Minh  {}'.format(len(total_train)+len(total_test)))
    # train, test = train_test_split(data_ducanh, test_size=0.2, random_state=3255)
    # total_train = total_train + train
    # total_test = total_test + test
    print('tổng dữ liệu của anh Minh và Đức Anh  {}'.format(len(train)+len(test)))
    # ghi vao file train va test o dang list de dua vao model
    with open(path+final_json[0], 'w') as outfile:
        json.dump(train, outfile, ensure_ascii=False)
    with open(path+final_json[1], 'w') as outfile:
        json.dump(test, outfile, ensure_ascii=False)

def get_statistical_erorr(final_erorr, erorr):
    final_erorr[0] = erorr[0] + erorr[13]
    final_erorr[1] = erorr[1] + erorr[14]
    final_erorr[2] = erorr[2] + erorr[15]
    final_erorr[3] = erorr[3] + erorr[16]
    final_erorr[4] = erorr[4]
    final_erorr[5] = erorr[5]
    final_erorr[6] = erorr[6]
    final_erorr[7] = erorr[7]
    final_erorr[8] = erorr[8] + erorr[17] + erorr[18]
    final_erorr[9] = erorr[9] + erorr[19] + erorr[20]
    final_erorr[10] = erorr[10] + erorr[21]
    final_erorr[11] = erorr[11] + erorr[22] + erorr[23]
    final_erorr[12] = erorr[12] + erorr[24] + erorr[25]
    return final_erorr

def get_length_data_json(fil):
    '''
     trả về kích thức dữ liệu gốc của anh Minh
    '''
    data = []
    with open(fil, 'r') as json_data:
        for element in json_data:
            data.append(json.loads(element))
    return len(data)

def get_length_data_add(file_ducanh):
    """
        trả về kích thước dữ liệu gốc của Đức Anh 
    """
    data = read_data_ducanh(file_ducanh)
    return len(data)

import collections
def random_numbers_sequence(path):
    # đọc dữ liệu 
    with open(path) as f:
        data = json.load(f)
    # sampling = random.choices(data, k = 1000)
    # for i in sampling:
    #     j3 = 0
    #     for j1 in i['original']:
    #         for k1 in j1: 
    #             if k1 in string.punctuation and k1 != '_' :
    #                 print('{:<10}:  {:<20}  {:<20}'.format(j3, i['id'],j1))
    #                 break
    #         j3 += 1
    # data1 = []
    # for i in data:
    #     data1.append(i['id'])
    # print(len(data1))
    # data2 = collections.Counter(data1)
    # print(data2)


    # test = ['NEWS_08156250','STORS_0232603','DUCANH__7063','DUCANH_71260','DUCANH_6810']
    test = ['BOOK_1005503MANH2', 'BOOK_1005503MANH1', 'BOOK_1005503MANH0', 'BOOK_1005503']

    for i in data:
        if i['id'] in test:
            # print(i['original'])
            print(i)

if __name__ == "__main__":
    f = None
    path = './data_test/'
    # tokenize_data = get_data_book(path + 'test_book.json')
    # noise_data = add_noise_sequen(tokenize_data, f)
    # data = shuffle(noise_data)
    # with open(path+'test_book1.json', 'w') as outfile:
    #     json.dump(data, outfile, ensure_ascii=False)

    random_numbers_sequence('./data_8_2/train_data.json')


    # câu dưới trong dữ liệu tự động thêm dấu _ vào sau từ  Bhưới sau khi áp dung tokenize 
    # a = 'Đúng như lời chủ tịch Đinh và già làng Bhling Chrlâng , qua những thôn làng ở các xã vùng biên này , giờ đây tôi đã gặp khá nhiều những thợ mộc địa phương , trong đó có rất nhiều thợ trẻ như thợ cả Jơrum Xia , 22 tuổi , là bí thư chi đoàn thôn Voòng ( xã Tr’hy ) , như Cơlâu Nhới , mới 17 tuổi - con của trưởng thôn Cơlâu Bhưới ...'

    # a = ViTokenizer.tokenize('Oắt!')
    # a = a.split(" ")
    # print(a)

    a = {'id': 'BOOK_0834713', 'raw': ['“', 'Có', 'lần', ',', 'trước', 'một', 'đám', 'đông', ',', 'ông', 'yêu_cầu', 'mọi', 'người', 'cởi', 'bỏ', 'âu_phục', 'để', 'ông', 'thiêu_hủy', '.'], 'original': ['“', 'Có', 'lần', ',', 'trước', 'một', 'đám', 'đông', ',', 'ông', 'yêu_cầu', 'mọi', 'người', 'cởi', 'bỏ', 'âu_phục', 'để', 'ông', 'thiêu_hủy', '.'], 'tid': 0}
    b = {'id': 'BOOK_08347113', 'raw': ['“', 'Có', 'lần', ',', 'trước', 'một', 'đám', 'đông', ',', 'ông', 'yêu_cầu', 'mọi', 'người', 'cởi', 'bỏ', 'âu_phục', 'để', 'ông', 'thiêu_hủy', '.'], 'original': ['“', 'Có', 'lần', ',', 'trước', 'một', 'đám', 'đông', ',', 'ông', 'yêu_cầu', 'mọi', 'người', 'cởi', 'bỏ', 'âu_phục', 'để', 'ông', 'thiêu_hủy', '.'], 'tid': 0}

    if a == b:
        print('1')