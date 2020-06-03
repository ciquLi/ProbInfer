#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:57:01 2019

@author: lixionglve
"""

import json
import pandas as pd
import os
import ujson
import pickle
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import Pool

current_path = os.getcwd()  #返回当前文件所在的目录
parent_path = os.path.dirname(current_path) #获得current_path所在的目录,即父级目录
files_path = parent_path + '/files/'    #获得存放数据的根目录

#获取后缀为file_type文件的路径
def get_dir(path,file_list,file_type):
    for file in os.listdir(path):
        file_path = path + '/' +file
        if os.path.isdir(file_path):
            if file[0] != '.':
                get_dir(file_path,file_list,file_type)
        else:
            if os.path.splitext(file_path)[1]==file_type:
                #获取当前文件所在目录名以及当前文件的相对路径,文件大小
                #father_path = os.path.dirname(file_path)
                #parent_name = os.path.split(father_path)[1]
                size = os.path.getsize(file_path)
                file_list.append([size,file_path])
    file_list.sort(reverse=True)    #按文件大小降序
                
                



def drop_duplicated_aspath():
    file_type = '.csv'
    path = files_path + 'data/aspath/test/path'
    file_list = []
    get_dir(path, file_list, file_type)
    for row in file_list:
        as_file = row[1]
        print(as_file)
        df = pd.read_csv(as_file, names=['aspath'])
        df_drop_dup = df.drop_duplicates()
        df_drop_dup.to_csv(as_file, index=False, header=None)


def SeparateASpath(current_dir):    #将一条到某个前缀的含有K个AS的路径分解成K-1条到该前缀的路径，每条路径至少有两个AS
    bgptotal = current_dir + '/BGPTotal.csv'
    new_aspath_file = current_dir + '/new_aspath.txt'
    intersection_prefix_file = current_dir + '/common_prefix/intersection_prefix.json'
    fp = open(intersection_prefix_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    intesection_prefix = ujson.loads(str_js)    #key为AS,值为拥有的前缀
    fp.close()
    
    if os.path.exists(new_aspath_file):
        os.remove(new_aspath_file)
    
    new_aspath = []
    #ground_path = []
    df = pd.read_csv(bgptotal,names=['aspath'])  
    for v in df.values:
        aspath_list = v[0].split('->')
        as_num = len(aspath_list)-1    #as的个数
        if as_num < 2:    #只用长度大于等于2的路径来生成确定路径
            continue
        count = 0
        stop = False    #如果as出现在了intersection_prefix的key中，则变为True,同时，抛弃该路径
        for asn in aspath_list[0:-2]:
            try:
                intesection_prefix[asn]
                #pos = aspath_list.index(asn)
                #gpath = aspath_list[pos:]
                #ground_path.append('->'.join(gpath))
                stop = True
                count += 1
                break
            except:
                continue
        if stop:
            continue

        for i in range(as_num):    #得到的每条路径至少有一个AS
            new = '->'.join(aspath_list[i:])
            new_aspath.append(new)

        if len(new_aspath) >= 1000000:
            df_new = pd.DataFrame(new_aspath)
            df_new.to_csv(new_aspath_file,index=False,header=None,mode='a')
            new_aspath = []
    df_new = pd.DataFrame(new_aspath)
    df_new.to_csv(new_aspath_file,index=False,header=None,mode='a')
    print(count)

    
def get_no_dup_aspath(current_dir):
    new_aspath_file = current_dir + '/new_aspath.txt'
    no_dup_aspath_file = current_dir + '/new_aspath_drop_duplicated.txt'
    df = pd.read_csv(new_aspath_file,names=['aspath'])
    df_drop_dup = df.drop_duplicates()
    df_drop_dup.to_csv(no_dup_aspath_file,index=False,header=None) 
    
def get_frequency(current_dir):
     #求每条路径的频率
    duplicated_aspath_file = current_dir + '/new_aspath.txt'
    var_dir = current_dir + '/var'
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)
    frequency_file = var_dir + '/frequency.txt'
    df_aspath = pd.read_csv(duplicated_aspath_file,header=None)
    counts=df_aspath[0].value_counts()
    frequency_dict = counts.to_dict()    #将每条路径的频率存到字典中，便于快速读取
    print('get frequency successfully')
    f=open(frequency_file,'wb')
    pickle.dump(frequency_dict,f)
    f.close()

    left_split_path_file = current_dir + '/left_split_path.csv'  # left_split_path  不带前缀地址的路径
    left_frequency_file = var_dir + '/left_frequency.txt'
    df_left_split_path = pd.read_csv(left_split_path_file, header=None)
    counts = df_left_split_path[0].value_counts()
    left_frequency_dict = counts.to_dict()  # 将每条路径的频率存到字典中，便于快速读取
    print('get left_frequency successfully')
    f = open(left_frequency_file, 'wb')
    pickle.dump(left_frequency_dict, f)
    f.close()
    df_drop_dup = df_left_split_path.drop_duplicates()
    df_drop_dup.to_csv(left_split_path_file, index=False, header=None)
    

        
def get_common_dest_prefix(n,m,current_dir):    #拥有共同目的前缀的vantage as  ，n为想要的目的前缀数量,m 需要的vantage as的数量
    file_type = '.json'
    path = current_dir + '/vatangeAS2prefix'
    file_list = []
    get_dir(path, file_list, file_type)
    #获取每个前缀拥有的路径数量
    prefix_path_num_file = current_dir + '/prefix_path_num.json'
    fp = open(prefix_path_num_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    prefix_path_num = ujson.loads(str_js)    #每个vantage AS可以到达的PREFIX
    #初始化common_prefix
    init_file = file_list[0][1]
    fp = open(init_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    vprefixs = ujson.loads(str_js)    #每个vantage AS可以到达的PREFIX
    common_prefix = list(vprefixs.values())[0]
    vantage_as = []    #存储可以到达共同前缀的vantage as
    sort_common_prefix = []
    for row in file_list:
        file = row[1]
        fp = open(file)
        str_js = ujson.load(fp)    #从文本中读取，str
        d = ujson.loads(str_js)    #每个vantage AS可以到达的PREFIX
        vprefixs = list(d.values())[0]
        #print(len(vprefixs),len(common_prefix))
        common_prefix = list(set(common_prefix).intersection(set(vprefixs)))
        vantage_as.append(list(d.keys())[0])    
        length = len(common_prefix)
        #print('number of common prefix:',length)
        #print('number of vantage as',len(vantage_as))
        if length <= n:
            if len(vantage_as) >= m:
                vantage_as = vantage_as[0:m]
            break
        if len(vantage_as) >= m:
            if length > n:
                for cp in common_prefix:
                    num = prefix_path_num[cp]
                    sort_common_prefix.append([num,cp])
                sort_common_prefix.sort(reverse=True)
                common_prefix = []
                for i in range(n):
                    common_prefix.append(sort_common_prefix[i][1])
            break
        
        
    return common_prefix,vantage_as
    
def get_path_by_common_prefix(prefix_list,vantage_as,current_dir):    #共同前缀来获取路径

    result = []    #以prefix_list中前缀为目的前缀的路径
    as_info = []    #记录每个vantage拥有的到达共同前缀的路径数量
    prefix_dict = {}    #构建共同前缀的字典，加快查找速度
    for prefix in prefix_list:
        prefix_dict[prefix] = 1
    for vas in vantage_as:
        path = current_dir + '/path/%s.csv'%(vas)
        #print(path)
        path_num = 0    #记录当前AS中到前缀地址集中地址的数量
        try:
            df = pd.read_csv(path,names=['aspath'])
        except:
            path = current_dir + '/otherpath/%s.csv'%(vas)
            print('get path by common prefix',vas)
            continue
        for value in df['aspath'].values.tolist():
            aslist = value.split('->')
            prefix = aslist[-1]
            try:
                prefix_dict[prefix]
                result.append(value)
                path_num += 1
            except:
                continue
        as_info.append([path_num,vas])
        #if len(result) >= 10000000:    #路径上限
        #    print('over 10000000')
         #   break
    return result,as_info

def split_common_prefix(common_prefix,common_vantage_as,current_dir):    #找到common_prefix属于的as 与剩下的vantage as 的交集
    #每个prefix属于的AS,路径全部经过筛选，不存在属于多个AS的前缀
    prefix_as_path = current_dir + '/prefix_as.json'
    fp = open(prefix_as_path)
    str_js = ujson.load(fp)    #从文本中读取，str
    prefix_as = ujson.loads(str_js)    #PREFIX属于的AS
    fp.close()
    #获取所有vantage as
    vantage_as_path = current_dir + '/ori_as.json'
    fp = open(vantage_as_path)
    str_js = ujson.load(fp)    #从文本中读取，str
    all_vantage_as = ujson.loads(str_js)    #PREFIX属于的AS
    fp.close()

    other_as = []    #除去common_vantage_as以外的观测点
    for asn in all_vantage_as.keys():
        if asn not in common_vantage_as:
            other_as.append(asn)
 
    common_as = []    #所有common_prefix属于的AS的集合
    for prefix in common_prefix:
        asn = prefix_as[prefix][0]
        common_as.append(asn)
    common_as = list(set(common_as))
    print('number of ASes that common prefixes belong to: ',len(common_as))
    intersection_as = list(set(common_as).intersection(set(other_as)))    #求交集，全部为vantage as
    dest_prefix = {}    #common_prefix除去属于intersection_as的前缀,存为字典格式为了判断是否属于时速度快
    intersection_prefix = {}    #每个intersection_as中的AS拥有的属于common_prefix的前缀
    for prefix in common_prefix:
        asn = prefix_as[prefix][0]
        if asn not in intersection_as:
            dest_prefix[prefix] = 1
        else:
            try:
                tmp = intersection_prefix[asn]
                tmp.append(prefix)
                intersection_prefix[asn] = tmp
            except:
                intersection_prefix[asn] = [prefix]
    return intersection_prefix,dest_prefix,intersection_as    #intersection_prefix+dest_prefix = common_prefix

def get_ground_path(intersection_as,dest_prefix,current_dir):    #dest_prefix = common_prefix - 属于intersection as 的前缀
    #####求intersection_as到dest_prefix的路径
    ground_path = []
    left_split_path = []    #intersection_as出发的，非ground_path的路径
    for ias in intersection_as:
        path = current_dir + '/path/%s.csv'%(ias)
        #print(path)
        try:
            df = pd.read_csv(path,names=['aspath'])
        except FileNotFoundError:
            #print(ias)
            continue
        for value in df['aspath'].values.tolist():
            aslist = value.split('->')
            #if len(aslist) <= 4:
            #    continue
            prefix = aslist[-1]
            try:
                dest_prefix[prefix]
                ground_path.append(value)
            except:
                aspath = '->'.join(aslist[0:-1])
                left_split_path.append(aspath)
    #left_split_path = list(set(left_split_path))
    return ground_path,left_split_path

def cut_common_prefix(cut,common_prefix,dest_prefix):
    intersection_prefix = []
    other_prefix = list(dest_prefix.keys())
    for prefix in common_prefix:
        try:
            dest_prefix[prefix]
        except:
            intersection_prefix.append(prefix)
            
    length= int(len(other_prefix)/cut)
    cut_prefix = other_prefix[0:length] + intersection_prefix
    return cut_prefix


   
def get_data(current_dir):    #随机找N个前缀，将可以到达这些前缀的路径全部找到
    n = 90000    #共同前缀数量上限
    m = 150    #vantage as 的上限

    run = True
    while run:
        common_prefix,vantage_as = get_common_dest_prefix(n,m,current_dir)
        intersection_prefix,dest_prefix,intersection_as = split_common_prefix(common_prefix,vantage_as,current_dir)    #intersection_prefix,dest_prefix都是字典
        ground_path,left_split_path = get_ground_path(intersection_as,dest_prefix,current_dir)
        result,as_info = get_path_by_common_prefix(common_prefix,vantage_as,current_dir)
        print('Number of common prefix:',len(common_prefix))
        print('Number of dest_prefix:',len(dest_prefix))
        print('Number of intersection as:',len(intersection_as))
        print('Number of vantage as:',len(as_info))
        print('Number of path to common prefix',len(result))
        print('Number of ground path',len(ground_path))

        if len(result) <= 30000:
            n = n + 10000
        else:
            run = False
    '''
    run = True
    cut = 1    #截取common_prefix
    while run:
        if len(result) >= 9000000:
            cut += 0.5
            cut_prefix = cut_common_prefix(cut,common_prefix,dest_prefix)    #按比例缩小
            result,as_info = get_path_by_common_prefix(cut_prefix,vantage_as,current_dir)
            print('Number of path to common prefix',len(result))
        else:
            run = False
    print('n: ',n)
    print('cut: ',cut)
    '''
    df_prefix = pd.DataFrame(common_prefix)
    prefix_dir = current_dir + '/common_prefix'
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    prefix_file = prefix_dir + '/common_prefix.txt'  #共同目的前缀
    df_prefix.to_csv(prefix_file, index=False, header=None)
    
    df_path =  pd.DataFrame(result)
    bgp_path = current_dir + '/BGPTotal.csv'  # vantage as到共同前缀的所有路径
    df_path.to_csv(bgp_path, index=False, header=None)
    
    df_groud_path =  pd.DataFrame(ground_path)
    ground_path_file = current_dir + '/common_prefix/ground_path.csv'  # ground path
    df_groud_path.to_csv(ground_path_file, index=False, header=None)

    df_left_split_path = pd.DataFrame(left_split_path)
    left_split_path_file = current_dir + '/left_split_path.csv'  # left_split_path  不带前缀地址的路径
    df_left_split_path.to_csv(left_split_path_file, index=False, header=None)
    
    
    vantage_as_file = current_dir + '/common_prefix/vantage_as.txt'  # 所有vantage as和拥有的路径条数
    df_as_info =  pd.DataFrame(as_info)
    df_as_info.to_csv(vantage_as_file, index=False, header=None)
    
    intersection_prefix_file = current_dir + '/common_prefix/intersection_prefix.json'
    fp = open(intersection_prefix_file, 'w')
    json_str = json.dumps(intersection_prefix)
    json.dump(json_str, fp)
    fp.close()
       
def split_data(test_ratio,current_dir):
    ground_path_file = current_dir + '/common_prefix/ground_path.csv'  # ground path
    df = pd.read_csv(ground_path_file,names=['aspath'])
    df_train,df_test = train_test_split(df,test_size=test_ratio)
    validate_dir = current_dir + '/bgp_validate'
    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)
    train_file = validate_dir + '/train.csv'
    df_train.to_csv(train_file, index=False, header=None)
    test_file = validate_dir + '/test.csv'
    df_test.to_csv(test_file, index=False, header=None)

def delete_var(current_dir):
    var_dir = current_dir + '/var/'
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)
        return
    file_list = os.listdir(var_dir)
    for file in file_list:
        if file != '.DS_Store':
            path = var_dir + file
            os.remove(path)

def multi_process(function,argc):
    # 多线程下载
    p = Pool(3)
    p.map(function, argc)
    p.close()
    p.join()

def child(para):
    file = para[0]
    test_ratio = para[1]
    if file == '.DS_Store':
        return
    print(file)
    root_dir = files_path + 'data/aspath/path_by_vantage/'
    current_dir = root_dir + file
    get_data(current_dir)
    delete_var(current_dir)
    split_data(test_ratio, current_dir)
    SeparateASpath(current_dir)
    get_no_dup_aspath(current_dir)
    get_frequency(current_dir)

def main(test_ratio):
    print('get_ground_path')
    root_dir = files_path + 'data/aspath/path_by_vantage/'
    vantage_aspath_list = os.listdir(root_dir)
    para = []
    for file in vantage_aspath_list:
        para.append([file,test_ratio])
    multi_process(child,para)





    
    


