#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:52:28 2019

@author: lixionglve
"""

from mrtparse import *
import os
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import json
import networkx as nx


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
                #获取当前文件所在目录名以及当前文件的相对路径
                father_path = os.path.dirname(file_path)
                date = os.path.split(father_path)[1]
                file_list.append([date,file_path])
                

def multi_process(function,argc,num_thread):
    # 多线程下载
    p = Pool(num_thread)
    p.map(function, argc)
    p.close()
    p.join()  
    
    

def read_bgp(para):    #从bgp数据中读取路径信息并保存  format:as1-as2-as3-prefix  para为数组，第一个参数为bgp文件地址，第二个为结果的存放地址
    filename = para[0]
    print(filename)
    save_path = para[1]
    data = Reader(filename)
    count = 0
    as2prefix_path = []    #自治域到前缀的路径
    for m in data:
        try:
            m = m.mrt
            if m.err:
                continue
            elif m.type == MRT_T['TABLE_DUMP_V2']:
                if (m.subtype == TD_V2_ST['RIB_IPV4_UNICAST']
                or m.subtype == TD_V2_ST['RIB_IPV4_MULTICAST']):
                    nlri = '%s/%d' % (m.rib.prefix, m.rib.plen)
                    for entry in m.rib.entry:
                        for attr in entry.attr:
                            if attr.as_path != None:
                               aspath = attr.as_path[0]['val']
                               state = False    #判断asn是否大于1000000
                               for asn in aspath:
                                   if int(asn) >= 1000000:
                                       state = True
                                       break
                               if state == True:
                                   continue

                               drop_duplicated_aspath = list(set(aspath))
                               if len(drop_duplicated_aspath) != len(aspath):
                                   continue

                               if len(aspath) < 2:
                                   continue
                               aspath.append(nlri)
                               str_aspath = '->'.join(aspath)
                               as2prefix_path.append(str_aspath)
                               count += 1
                               if (count%10000) == 0:
                                   df = pd.DataFrame(as2prefix_path)
                                   df.to_csv(save_path,index=False,header=None,mode='a')
                                   as2prefix_path = []
                                   #print(count)
        except:
            continue
                          
    df = pd.DataFrame(as2prefix_path)
    df.to_csv(save_path,index=False,header=None,mode='a')
    print('total aspath number:',count)
                

def multi_process_GetASPathFromBGP(filetype):    #多线程处理BGP文件
    bgp_path = files_path + 'data/raw_data/bgp' 
    
    file_list = []
    get_dir(bgp_path,file_list,filetype)
   # print(file_list)
    aspath = files_path + 'data/aspath/bgp/'
    para = []
    for f in file_list:
        file = f[1]
        filename = file.split('/')[-1][:-len(filetype)]
        father_dir = f[0]
        save_dir =  aspath + father_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/' + filename + '.csv'
        if not os.path.exists(save_path):
            para.append([file,save_path])
    lock = multiprocessing.Lock()
    multi_process(read_bgp,para,num_thread=40)

def get_basic_info(parent):
    file_type = '.csv'
    path = files_path + 'data/aspath/bgp/' + parent
    file_list = []
    get_dir(path, file_list, file_type)
    ori_as = {}    #存储源AS
    prefix_as = {}    #每个前缀属于的AS
    as_prefix = {}    #每个AS拥有的前缀
    for row in file_list:
        file = row[1]
        #print(file)
        df = pd.read_csv(file, names=['aspath'])
        for v in df['aspath'].values:
            as_list = v.split('->')
            start_as = as_list[0]
            ori_as[start_as] = start_as
            prefix = as_list[-1]
            last_as = as_list[-2]
            try:
                tmp = as_prefix[last_as]
                if prefix not in tmp:
                    tmp.append(prefix)
            except:
                as_prefix[last_as] = [prefix]
            try:
                tmp = prefix_as[prefix]
                if last_as not in tmp:
                    tmp.append(last_as)
            except:
                prefix_as[prefix] = [last_as]
        #print(len(ori_as))
    moas = []    #multi original as
    soas = []    #single original as
    for key in prefix_as.keys():
        if len(prefix_as[key]) > 1:
            moas.append(key)
        else:
            soas.append(key)
            
    result_dir = files_path + 'data/aspath/path_by_vantage/' + parent
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df_moas = pd.DataFrame(moas)
    df_soas = pd.DataFrame(soas)
    moas_file = result_dir + '/moas.txt'
    df_moas.to_csv(moas_file,index=False,header=None)
    soas_file = result_dir + '/soas.txt'
    df_soas.to_csv(soas_file,index=False,header=None)

    ori_as_path = result_dir + '/ori_as.json'
    fp = open(ori_as_path, 'w')
    json_str = json.dumps(ori_as)
    json.dump(json_str, fp)
    fp.close()
    prefix_as_path = result_dir + '/prefix_as.json'
    fp = open(prefix_as_path, 'w')
    json_str = json.dumps(prefix_as)
    json.dump(json_str, fp)
    fp.close()
    as_prefix_path = result_dir + '/as_prefix.json'
    fp = open(as_prefix_path, 'w')
    json_str = json.dumps(as_prefix)
    json.dump(json_str, fp)
    fp.close()

def get_path_by_as(parent):
    file_type = '.csv'
    path = files_path + 'data/aspath/bgp/' + parent
    file_list = []
    get_dir(path, file_list, file_type)
    result_dir = files_path + 'data/aspath/path_by_vantage/' + parent
    as_dir = result_dir + '/path'
    if not os.path.exists(as_dir):
        os.makedirs(as_dir)
    moas_file = result_dir + '/moas.txt'
    df_moas = pd.read_csv(moas_file, names=['moas'])
    moas = df_moas['moas'].values.tolist()
    moas_dict = {}
    for mprefix in moas:
        moas_dict[mprefix] = mprefix
    for row in file_list:
        file = row[1]
        #print(file)
        df = pd.read_csv(file, names=['aspath'])
        path_dict = {}
        for v in df['aspath'].values:
            as_list = v.split('->')
            start_as = as_list[0]
            prefix = as_list[-1]
            #last_as = as_list[-2]
            try:
                mp = moas_dict[prefix]    #如果一个前缀对应多个源AS，则舍弃这条路径
                continue
            except:
                try:
                    tmp = path_dict[start_as]
                    tmp.append(v)
                except:
                    path_dict[start_as] = [v]
        
        for key in path_dict.keys():
            tmp = path_dict[key]
            
            as_file = as_dir + '/' + key + '.csv'
            df_tmp = pd.DataFrame(tmp)
            df_tmp.to_csv(as_file, index=False, header=None, mode='a')

def get_path_by_prefix(parent):    #到每个前缀的路径条数
    prefix_path_num  = {}   
    file_type = '.csv'
    path = files_path + 'data/aspath/path_by_vantage/' + parent + '/path'
    file_list = []
    get_dir(path, file_list, file_type)
    for row in file_list:
        file = row[1]
        #print(file)
        df = pd.read_csv(file, names=['aspath'])
        for v in df['aspath'].values:
            as_list = v.split('->')
            prefix = as_list[-1]
            try:
                num = prefix_path_num[prefix]
                num += 1
                prefix_path_num[prefix] = num
            except:
                prefix_path_num[prefix] = 1
    prefix_path_num_file = files_path + 'data/aspath/path_by_vantage/' + parent + '/prefix_path_num.json'
    fp = open(prefix_path_num_file, 'w')
    json_str = json.dumps(prefix_path_num)
    json.dump(json_str, fp)
    fp.close()
            
def drop_duplicated_aspath(parent):
    file_type = '.csv'
    path = files_path + 'data/aspath/path_by_vantage/' + parent + '/path'
    file_list = []
    get_dir(path, file_list, file_type)
    for row in file_list:
        as_file = row[1]
        #print(as_file)
        df = pd.read_csv(as_file, names=['aspath'])
        df_drop_dup = df.drop_duplicates()
        df_drop_dup.to_csv(as_file, index=False, header=None)

def analyze_data(parent):
    file_type = '.csv'
    path = files_path + 'data/aspath/path_by_vantage/' + parent + '/path'
    file_list = []
    get_dir(path, file_list, file_type)
    prefix_dict = {}
    asn_dict = {}
    aspath_num = 0
    for row in file_list:
        path_file = row[1]
        print(path_file)
        df = pd.read_csv(path_file, names=['aspath'])
        for v in df['aspath'].values:
            as_list = v.split('->')
            prefix = as_list.pop(-1)
            prefix_dict[prefix] = prefix
            for asn in as_list:
                asn_dict[asn] = asn
        aspath_num += len(df)

        print('Number of prefix:', len(prefix_dict))
        print('Number of asn:', len(asn_dict))
        print('Number of aspath:', aspath_num)
    analyze_dir = files_path + 'data/aspath/path_by_vantage/' + parent + '/analyze'
    if not os.path.exists(analyze_dir):
        os.makedirs(analyze_dir)
    prefix_file = analyze_dir + '/prefix.json'
    asn_file = analyze_dir + '/asn.json'
    analyze_file = analyze_dir + '/analyze.txt'
    fp = open(prefix_file, 'w')
    json_str = json.dumps(prefix_dict)
    json.dump(json_str, fp)
    fp.close()
    fp = open(asn_file, 'w')
    json_str = json.dumps(asn_dict)
    json.dump(json_str, fp)
    fp.close()
    with open(analyze_file, 'w') as fp:
        fp.write('Number of prefix:%d\n' % len(prefix_dict))
        fp.write('Number of asn:%d\n' % len(asn_dict))
        fp.write('Number of aspath:%d\n' % aspath_num)

def vantageAS2prefix(parent):    #每个vantage AS可以到达的前缀
    file_type = '.csv'
    path = files_path + 'data/aspath/path_by_vantage/' + parent + '/path'
    file_list = []
    get_dir(path, file_list, file_type)
    as2prefix = {}    #每个VANTAGE AS可到达的前缀
    for row in file_list:
        file = row[1]
        #print(file)
        df = pd.read_csv(file, names=['aspath'])
        for v in df['aspath'].values.tolist():
            nodes = v.split('->')
            prefix = nodes[-1]
            vantage_as = nodes[0]
            try:
                tmp = as2prefix[vantage_as]
                tmp[prefix] = 1
                as2prefix[vantage_as] = tmp
            except:
                tmp = {}
                tmp[prefix] = 1
                as2prefix[vantage_as] = tmp
    
    for asn in as2prefix.keys():
        tmp = as2prefix[asn]
        prefixs = list(tmp.keys())
        vantageAS2prefix = {}
        vantageAS2prefix[asn] = prefixs        
        vantage_as2prefix_dir = files_path + 'data/aspath/path_by_vantage/' + parent + '/vatangeAS2prefix' 
        if not os.path.exists(vantage_as2prefix_dir):
            os.makedirs(vantage_as2prefix_dir)
        vantage_as2prefix = vantage_as2prefix_dir +  '/%s.json'%(asn)
        fp = open(vantage_as2prefix, 'w')
        json_str = json.dumps(vantageAS2prefix)
        json.dump(json_str, fp)
        fp.close()
        
def prefix2as():
    prefix2as_file = files_path + 'data/aspath/prefix2as.txt'
    datafile = open(prefix2as_file, mode='r')
    prefix2as = datafile.read().splitlines()
    datafile.close()
    prefix2as_json = prefix2as_file = files_path + 'data/aspath/prefix2as.json'
    pre2as_dict = {}
    for row in prefix2as:
        row = row.split(' ')
        prefix = row[0] + '/' + row[1]
        asn = row[2]
        pre2as_dict[prefix] = asn
    fp = open(prefix2as_json, 'w')
    json_str = json.dumps(pre2as_dict)
    json.dump(json_str, fp)
    fp.close()



def get_path_without_prefix(parent):
    file_type = '.csv'
    path = files_path + 'data/aspath/path_by_vantage/' + parent + '/path'
    file_list = []
    get_dir(path, file_list, file_type)
    result_dir = files_path + 'data/aspath/path_by_vantage/' + parent
    as_dir = result_dir + '/path_without_prefix'
    if not os.path.exists(as_dir):
        os.makedirs(as_dir)
    
    for row in file_list:
        file = row[1]
        print(row)
        df = pd.read_csv(file, names=['aspath'])
        tmp = []
        
        for v in df['aspath'].values:
            as_list = v.split('->')
            start_as = as_list[0]
            #prefix = as_list[-1]
            #end_as = as_list[-2]
            aspath = '->'.join(as_list[0:-1])
            tmp.append(aspath)
 
        tmp = list(set(tmp))
        as_file = as_dir + '/' + start_as + '.csv'
        df_tmp = pd.DataFrame(tmp)
        df_tmp.to_csv(as_file, index=False, header=None, mode='a')
    
    #去重
    file_type = '.csv'
    file_list = []
    get_dir(as_dir, file_list, file_type)
    for row in file_list:
        as_file = row[1]
        print(as_file)
        df = pd.read_csv(as_file, names=['aspath'])
        df_drop_dup = df.drop_duplicates()
        df_drop_dup.to_csv(as_file, index=False, header=None)
        

def child(file):
    parent = file
    if parent == '.DS_Store':
        return
    print(parent)

    prefix2as()
    print('get_basic_info')
    get_basic_info(parent)
    print('get_path_by_as')
    get_path_by_as(parent)
    print('drop_duplicated_aspath')
    drop_duplicated_aspath(parent)
    print('get_path_by_prefix')
    get_path_by_prefix(parent)
    print('vantageAS2prefix')
    vantageAS2prefix(parent)
    get_path_without_prefix(parent)


def main():
    filetypes = ['.bz2', '.gz']
    # filetypes = ['.gz']
    for filetype in filetypes:
        multi_process_GetASPathFromBGP(filetype)

    aspath = files_path + 'data/aspath/bgp'
    file_list = os.listdir(aspath)
    multi_process(child, file_list, num_thread=4)


        
# if __name__=="__main__":

#     aspath = files_path + 'data/aspath/bgp'
#     file_list = os.listdir(aspath)
#     for file in file_list:
#         parent = file
#         if parent == '.DS_Store':
#             continue
#         print(parent)
#         #prefix2as()
#         print('get_basic_info')
#         #get_basic_info(parent)
#         print('get_path_by_as')
#         #get_path_by_as(parent)
#         print('drop_duplicated_aspath')
#         #drop_duplicated_aspath(parent)
#         print('get_path_by_prefix')
#         #get_path_by_prefix(parent)
#         print('vantageAS2prefix')
#         #vantageAS2prefix(parent)
#         print('get_path_without_prefix')
#         #get_path_without_prefix(parent)
#         print('global_topo')
#         global_topo(parent)
#         get_country_topo(parent)
#         analyze_AStopo(parent)
    
    