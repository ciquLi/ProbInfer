#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:49:24 2019

@author: lixionglve
"""

import pandas as pd
import os
import ujson
import pickle
import networkx as nx

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
                #获取当前文件所在目录名以及当前文件的相对路径
                father_path = os.path.dirname(file_path)
                date = os.path.split(father_path)[1]
                file_list.append([date,file_path])




def get_edges(current_dir):
    aspath_file = current_dir + '/new_aspath_drop_duplicated.txt'
    edges = []
    df = pd.read_csv(aspath_file,names=['aspath'])
    for path in df['aspath'].values:
        as_list = path.split('->')
        start_as = as_list[0]
        prefix = as_list[-1]
        edge = [start_as,prefix]
        edges.append(edge)

    df_edges = pd.DataFrame(edges).drop_duplicates()
    edge_list = df_edges.values.tolist()
    return edge_list

def get_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    print('create graph successfully')
    return G


def multi_process(function,argc,num_process):
    # 多进程
    p = Pool(num_process)
    result = p.map(function, argc)
    p.close()
    p.join()
    return result

def get_cn(para):
    keys = para[0]
    ground_path = para[1]
    G = para[2]
    as_prefix = para[3]
    cn_dict = {}
    tmp_ground_path ={}

    print('key',len(keys))
    count = 0
    for key in keys:
        pair = key.split('-')
        start_as = pair[0]
        start_prefixs = as_prefix[start_as]
        end = pair[1]
        status = True
        for start in start_prefixs:
            try:
                cn = list(nx.common_neighbors(G, start, end))
                if len(cn) != 0:
                    prefix_pair = start + '-' + end
                    cn_dict[prefix_pair] = cn
                    tmp_ground_path[prefix_pair] = ground_path[key]
                    status = False
                    break
            except :  # 图中没有start或者end节点
                continue
        
        if status:
            prefix_pair = start_prefixs[0] + '-' + end
            tmp_ground_path[prefix_pair] = ground_path[key]
            cn_dict[prefix_pair] = []
            count += 1
    print('cn_dict',len(cn_dict)-count)
    return [cn_dict,tmp_ground_path]


def get_common_neighbor(G,groud_path,as_prefix,save_dir):  # 返回prefix pair的共同邻居
    n = 4  # 进程数
    prefix_pair = list(groud_path.keys())
    length = len(prefix_pair)
    seg = length // n
    li = []
    for i in range(n - 1):
        tmp = prefix_pair[i * seg:(i + 1) * seg]
        li.append([tmp,groud_path,G,as_prefix])
    tmp = prefix_pair[(n - 1) * seg:]
    li.append([tmp,groud_path,G,as_prefix])

    result = multi_process(get_cn,li,n)
    cn_dict = {}
    g_path = {}
    for r in result:
        cn_dict.update(r[0])
        g_path.update(r[1])

    cn_file = save_dir + '/common_neighbors.txt'
    f=open(cn_file,'wb')
    pickle.dump(cn_dict,f)
    f.close()
    
    print(len(g_path))
    ground_path_file = save_dir + '/ground_path.txt'
    f=open(ground_path_file,'wb')
    pickle.dump(g_path,f)
    f.close()
    
    #fp = open(cn_file, 'w')
    #json_str = ujson.dumps(cn_dict)
    #ujson.dump(json_str, fp)
    #fp.close()


def main():
    root_dir = files_path + 'data/aspath/path_by_vantage/'
    vantage_aspath_list = os.listdir(root_dir)
    for file in vantage_aspath_list:
        if file == '.DS_Store':
            continue
        current_dir = root_dir + file
        child(current_dir)
        
def child(current_dir):
    print('get validate data')
    edges = get_edges(current_dir)
    G = get_graph(edges)
    #得到每个vantage as 拥有的属于common_prefix的前缀
    intersection_prefix_file = current_dir + '/common_prefix/intersection_prefix.json'
    fp = open(intersection_prefix_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    as_prefix = ujson.loads(str_js)    #每个AS拥有的PREFIX
    fp.close()
    
    file_type = '.csv'
    validate_bgp_dir = current_dir + '/bgp_validate'
    validate_bgp_file_list = []
    get_dir(validate_bgp_dir, validate_bgp_file_list, file_type)

    for row in validate_bgp_file_list:
        file = row[1]
        name = file.split('/')[-1][:-4]
        print(file)
        validate_dir = current_dir + '/validate'
        if not os.path.exists(validate_dir):
            os.makedirs(validate_dir)
        validate_dir_child = validate_dir + '/' + name
        if not os.path.exists(validate_dir_child):
            os.makedirs(validate_dir_child)

        df = pd.read_csv(file,names=['aspath'])
        print('initiate path number:',len(df))
        
        prefix_dict = {}
        validate_path = {}
        groud_path = {}
        for v in df['aspath'].values:
            nodes = v.split('->')
            end_prefix = nodes[-1]
            prefix_dict[end_prefix] = '1'
            start_as = nodes[0]
            try:
                start_prefixs = as_prefix[start_as]
            except:
                continue
            #for start_prefix in start_prefixs:
            #start_prefix = start_prefixs[0]
            #pair = start_prefix + '-' + end_prefix
            
            pair = start_as + '-' + end_prefix
            aspath = '->'.join(nodes[:-1])
            try:
                tmp = validate_path[pair]
                if aspath not in tmp:
                    tmp.append(aspath)
            except:
                validate_path[pair] =  [aspath]
        for key in validate_path.keys():
            tmp = validate_path[key]
            if len(tmp) == 1:
                groud_path[key] = tmp[0]

        prefix_pair = list(groud_path.keys())
        print('final ground path num: ',len(prefix_pair))

        get_common_neighbor(G,groud_path,as_prefix,validate_dir_child)

# if __name__=="__main__":
#     edges = get_edges()
#     G = get_graph(edges)
#     #得到每个vantage as 拥有的属于common_prefix的前缀
    
#     #as_prefix_path = files_path + 'data/aspath/test/as_prefix.json'
#     intersection_prefix_file = files_path + 'data/aspath/test/common_prefix/intersection_prefix.json'
#     fp = open(intersection_prefix_file)
#     str_js = ujson.load(fp)    #从文本中读取，str
#     as_prefix = ujson.loads(str_js)    #每个AS拥有的PREFIX
#     fp.close()
    
#     file_type = '.csv'
#     validate_bgp_dir = files_path + 'data/aspath/bgp_validate'
#     validate_bgp_file_list = []
#     get_dir(validate_bgp_dir, validate_bgp_file_list, file_type)

#     for row in validate_bgp_file_list:
#         file = row[1]
#         name = file.split('/')[-1][:-4]
#         print(file)
#         validate_dir = files_path + 'data/aspath/validate/' + name
#         if not os.path.exists(validate_dir):
#             os.makedirs(validate_dir)

#         df = pd.read_csv(file,names=['aspath'])
#         print('initiate path number:',len(df))
        
#         prefix_dict = {}
#         validate_path = {}
#         groud_path = {}
#         for v in df['aspath'].values:
#             nodes = v.split('->')
#             end_prefix = nodes[-1]
#             prefix_dict[end_prefix] = '1'
#             start_as = nodes[0]
#             try:
#                 start_prefixs = as_prefix[start_as]
#             except:
#                 continue
#             #for start_prefix in start_prefixs:
#             #start_prefix = start_prefixs[0]
#             #pair = start_prefix + '-' + end_prefix
            
#             pair = start_as + '-' + end_prefix
#             aspath = '->'.join(nodes[:-1])
#             try:
#                 tmp = validate_path[pair]
#                 if aspath not in tmp:
#                     tmp.append(aspath)
#             except:
#                 validate_path[pair] =  [aspath]
#         for key in validate_path.keys():
#             tmp = validate_path[key]
#             if len(tmp) == 1:
#                 groud_path[key] = tmp[0]

#         ground_file = validate_dir + '/ground_path.txt'
#         prefix_pair = list(groud_path.keys())
#         print('final ground path num: ',len(prefix_pair))

#         get_common_neighbor(G,groud_path,validate_dir)
        
        
        