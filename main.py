# -*- coding:utf-8 -*-
import re
import os
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# 初始化数据
def init_data():
    counter_ham = 0
    counter_spam = 0
    list_ham = []
    list_spam = []

    with open(data_file, 'r', encoding='UTF-8') as file:
        line = file.readline()
        while line:
            result = re.search(r'ham\t(.+)', line)
            if result is not None:
                # file_ham.write(result.group(1) + '\n')
                fil_line = re.sub(r'[.!#%^$@&*()_+<>?:"|{},`\'\-=/;\n]', '', line.lower())
                result = re.split(r' ', fil_line)
                for w in result:
                    if w != '':
                        list_ham.append(w)
                counter_ham += 1
            result = re.search(r'spam\t(.+)', line)
            if result is not None:
                # file_spam.write(result.group(1) + '\n')
                fil_line = re.sub(r'[.!#%^$@&*()_+<>?:"|{},`\'\-=/;\n]', '', line.lower())
                result = re.split(r' ', fil_line)
                for w in result:
                    if w != '':
                        list_spam.append(w)
                counter_spam += 1
            line = file.readline()

    list_spam = filters(list_spam)
    list_spam = stopword(list_spam)
    list_spam = lemmatization(list_spam)
    list_all = list_ham + list_spam

    return list_ham, list_spam, list_all, counter_ham, counter_spam


# 过滤
def filters(s_list):
    n_list = []
    for string in s_list:
        if len(string) > 1:
            result = re.search(r'([a-z]+)', string)
            if result is not None:
                n_list.append(result.group(1))
    return n_list


# 停用词处理
def stopword(s_list):
    for w in s_list:
        if w in stopwords.words('english'):
            s_list.remove(w)
    n_list = s_list
    return n_list


# 词干提取
def stemming(s_list):
    stemmer = PorterStemmer()
    n_list = []
    for w in s_list:
        w = stemmer.stem(w)
        n_list.append(w)
    return n_list


# 词形还原
def lemmatization(s_list):
    lemmatizer = WordNetLemmatizer()
    n_list = []
    for w in s_list:
        w = lemmatizer.lemmatize(w)
        n_list.append(w)
    return n_list


# 测试
def test():

    total = 0
    correct = 0
    # 先验概率
    p_ham = counter_ham / (counter_ham + counter_spam)
    p_spam = counter_spam / (counter_ham + counter_spam)

    # 后验概率
    with open(test_file, 'r', encoding='UTF-8') as file:
        line = file.readline()
        while line:
            total += 1
            pp_ham = p_ham
            pp_spam = p_spam
            flag = 0
            count = 0
            # 单词处理
            test_list = []
            fil_line = re.sub(r'[.!#%^$@&*()_+<>?:"|{},`\'\-=/;\n]', '', line.lower())
            result = re.split(r' ', fil_line)
            for w in result:
                test_list.append(w)
            test_list = filters(test_list)
            test_list = stopword(test_list)
            test_list = lemmatization(test_list)

            # 判断是否需要拉普拉斯平滑
            for s in test_list:
                if freq_ham.get(s) is None or freq_spam.get(s) is None:
                    count += 1
                    flag = 1

            # 计算后验概率
            if flag == 1:
                K = word_count + count
                for s in test_list:
                    freq_t_ham = float(freq_ham[s]) + 1 if freq_ham.get(s) is not None else 1
                    pp_ham *= freq_t_ham / (sum_ham + K)
                    freq_t_spam = float(freq_spam[s]) + 1 if freq_spam.get(s) is not None else 1
                    pp_spam *= freq_t_spam / (sum_spam + K)
            else:
                for s in test_list:
                    freq_t_ham = float(freq_ham[s])
                    pp_ham *= freq_t_ham / sum_ham
                    freq_t_spam = float(freq_ham[s])
                    pp_spam *= freq_t_spam / sum_spam

            # 预测结果
            predict_type = "ham" if pp_ham > pp_spam else "spam"

            # 输出结果
            result = re.search('ham\t(.+)', line)
            if result is not None:
                if predict_type == 'ham':
                    correct += 1
                print("Predicted：" + predict_type + "    Actual：ham")
                print("Text：" + result.group(1))
            result = re.search('spam\t(.+)', line)
            if result is not None:
                if predict_type == 'spam':
                    correct += 1
                print("Predicted：" + predict_type + "    Actual：spam")
                print("Text：" + result.group(1))
            print("-----------------------------------------------------")
            line = file.readline()
    print("FINISH! Total: " + str(total) + ". Hit: " + str(correct) + ". Accuracy: " + str(correct/total * 100) + "%.")


if __name__ == '__main__':

    data_file = 'SMSSpamCollection'
    test_file = 'test.txt'
    trained_file = 'trained.json'

    # 下载停用集与词汇网
    nltk.download('stopwords')
    nltk.download('wordnet')

    # 处理训练数据(如已有训练数据，则直接加载)
    if os.path.exists(trained_file):
        # 加载训练数据
        print("Found trained data. Loading...")
        with open(trained_file, 'r', encoding='UTF-8') as f:
            trained_dict = json.loads(json.load(f))
            freq_ham = json.loads(trained_dict['json_ham'])
            freq_spam = json.loads(trained_dict['json_spam'])
            counter_ham = float(trained_dict['counter_ham'])
            counter_spam = float(trained_dict['counter_spam'])
            sum_ham = float(trained_dict['sum_ham'])
            sum_spam = float(trained_dict['sum_spam'])
            word_count = float(trained_dict['word_count'])

    else:
        # 数据处理
        print("Data processing...")
        list_ham, list_spam, list_all, counter_ham, counter_spam = init_data()

        # 生成频率字典
        freq_ham = nltk.FreqDist(list_ham)
        freq_spam = nltk.FreqDist(list_spam)
        freq_all = nltk.FreqDist(list_all)
        sum_ham = freq_ham.N()
        sum_spam = freq_spam.N()
        word_count = freq_all.B()

        # 转化为json
        json_ham = json.dumps(freq_ham)
        json_spam = json.dumps(freq_spam)
        json_all = json.dumps(freq_all)

        # 保存训练数据
        trained_dict = {'sum_ham': sum_ham, 'sum_spam': sum_spam, 'word_count': word_count, 'counter_ham': counter_ham, 'counter_spam': counter_spam, 'json_ham': json_ham, 'json_spam': json_spam}
        trained_json = json.dumps(trained_dict)
        with open(trained_file, 'w', encoding='UTF-8') as f:
            json.dump(trained_json, f, ensure_ascii=False)

    test()
