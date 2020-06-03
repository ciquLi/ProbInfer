#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:53:08 2019

@author: lixionglve
"""

import pandas as pd
import numpy as np

import ujson

from multiprocessing import Pool
import os
import networkx as nx
import regular 

import pickle



current_path = os.getcwd()  #返回当前文件所在的目录   
parent_path = os.path.dirname(current_path) #获得current_path所在的目录,即父级目录 
files_path = parent_path + '/files/'    #获得存放数据的根目录




def get_multi_degree(edges):
    G = nx.MultiGraph()
    G.add_edges_from(edges)
    degree = list(G.degree())    #multigraph 中的度
    multi_degree = {}
    for l in degree:
        key = l[0]
        neighbor = l[1]
        multi_degree[key] = neighbor
    return multi_degree

def get_graph(edges):    #普通的图
    df_edges = pd.DataFrame(edges).drop_duplicates()
    edge_list = df_edges.values.tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G

def get_known_path(current_dir):
    aspath_file = current_dir + '/new_aspath_drop_duplicated.txt'
    left_split_path_file = current_dir + '/left_split_path.csv'  # left_split_path    不带前缀地址的路径
    edges = []
    known_aspath = {}
    left_split_path = {}
    df_left = pd.read_csv(left_split_path_file,names=['aspath'])
    for path in df_left['aspath'].values:
        as_list = path.split('->')
        start_as = as_list[0]
        end_as = as_list[-1]
        pair = start_as+'-'+end_as
        try:
            tmp = left_split_path[pair]
            tmp.append(path)
            left_split_path[pair] = tmp
        except:
            left_split_path[pair] = [path]
            
    df = pd.read_csv(aspath_file,names=['aspath'])
    for path in df['aspath'].values:
        as_list = path.split('->')
        start_as = as_list[0]
        prefix = as_list[-1]

        edge = [start_as,prefix]
        edges.append(edge)
        pair = start_as+'-'+prefix
        try:
            tmp = known_aspath[pair]
            tmp.append(path)
            known_aspath[pair] = tmp
        except:
            known_aspath[pair] = [path]
    return edges,known_aspath,left_split_path

def get_basic_info(current_dir):
    known_path_file = current_dir + '/var/known_path.txt'
    edges_file = current_dir + '/var/edges.txt'
    left_split_path_file = current_dir + '/var/left_split_path.txt'
    if os.path.exists(known_path_file) and os.path.exists(edges_file) and os.path.exists(left_split_path_file):
         f = open(known_path_file,'rb')
         known_path = pickle.load(f)
         f.close()
         f = open(left_split_path_file,'rb')
         left_split_path = pickle.load(f)
         f.close()
         f = open(edges_file,'rb')
         edges = pickle.load(f)
         f.close()
    else:
        edges,known_path,left_split_path = get_known_path(current_dir)
        f=open(known_path_file,'wb')
        pickle.dump(known_path,f)
        f.close()
        f=open(left_split_path_file,'wb')
        pickle.dump(left_split_path,f)
        f.close()
        f=open(edges_file,'wb')
        pickle.dump(edges,f)
        f.close()
    print('get known aspath successfully!')
    
    multi_degree_file = current_dir + '/var/multi_degree.txt'
    if os.path.exists(multi_degree_file):
         f = open(multi_degree_file,'rb')
         multi_degree = pickle.load(f)
         f.close()
    else:
        multi_degree = get_multi_degree(edges)
        f=open(multi_degree_file,'wb')
        pickle.dump(multi_degree,f)
        f.close()
    print('get multigraph degree successfully!')
    
    graph_file = current_dir + '/var/graph.txt'
    if os.path.exists(graph_file):
        f = open(graph_file,'rb')
        G = pickle.load(f)
        f.close()
    else:
        G = get_graph(edges)
        f=open(graph_file,'wb')
        pickle.dump(G,f)
        f.close()
    print('get graph successfully!')
    return multi_degree,G,known_path,left_split_path

    
def shortest_path(G,validate_aspath,cn_file,current_dir):
    #common neighbor
    f = open(cn_file,'rb')
    cn_dict = pickle.load(f)
    f.close()
    
    #每个prefix属于的AS,路径全部经过筛选，不存在属于多个AS的前缀
    prefix_as_path = current_dir + '/prefix_as.json'
    fp = open(prefix_as_path)
    str_js = ujson.load(fp)    #从文本中读取，str
    prefix_as = ujson.loads(str_js)    #PREFIX属于的AS
    fp.close()
    
    has_path = 0
    no_path = 0
    no_path_key = []
    one_seg_path = 0    #源、目的AS间存在真实路径
    one_seg_key = []
    two_seg_path = 0    #由两段拼接起来的
    two_seg_key = []

    other = 0    #可能是没路径，也可能是路径长度大于2
    other_seg_key = []
    #count = 0
    tmp = list(validate_aspath.keys())
    for key in tmp:
        prefix_pair = key.split('-')
        start = prefix_pair[0]
        end = prefix_pair[1]
        start_as = prefix_as[start][0]
        try:
            e = dict(G[start_as][end])
            one_seg_path += 1
            infer_path = [start,start_as,end]
            one_seg_key.append(key)
        except KeyError:
            try:
                cn = cn_dict[key]
                if len(cn) != 0:
                    infer_path = []
                    for node in cn:
                        infer_path.append([start,node,end])
                    two_seg_path += 1
                    two_seg_key.append([key,infer_path])
                else:
                    G[start]
                    G[end]
                    if nx.has_path(G, source=start, target=end):
                        other += 1
                        other_seg_key.append(key)
                    else:
                        no_path += 1
                        no_path_key.append(key)
            except KeyError:    #start或者end在图中不存在
                no_path += 1
                no_path_key.append(key)
#        count += 1       
        #if count%100 ==0:
         #   print(count)
    has_path = one_seg_path + two_seg_path + other
    print('validate_as_pair count',len(validate_aspath))
    print('has_path',has_path)
    print('no_path',no_path)
    print('one_seg_path',one_seg_path)
    print('two_seg_path',two_seg_path)
    print('other',other)

    result_file = current_dir + '/result.txt'
    fp = open(result_file, 'a')
    fp.write('total as pair:%d\n' % len(validate_aspath))
    fp.write('has path:%d\n' % has_path)
    fp.write('no path:%d\n' % no_path)
    fp.write('one seg:%d\n' % one_seg_path)
    fp.write('two seg:%d\n' % two_seg_path)
    fp.write('other seg:%d\n' % other)
    fp.close()

    return has_path,no_path_key,one_seg_key,two_seg_key,other_seg_key

############
'''
exist_path为一个list，为一个AS PAIR间的所有路径

'''
############
def sort_exist_aspath(exist_path,frequency_dict,neighbor_num_dict):
    sorted_path = []
    for path in exist_path:
        weight = compute_weight(path,frequency_dict,neighbor_num_dict)
        sorted_path.append([weight,path])
    sorted_path.sort(reverse = False)
    return sorted_path


def process_feature(data):    # 每条路径的属性在AS PAIR中所有路径中的排名
    array = np.array(data)
    m =  array.shape[0]    #行数
    n = array.shape[1]    #列数
    for i in range(n-4):    #不用规则化的列数
        feature = list(array[:,i])
        sort_feature = list(set(feature))
        if i < 1:    #特征按升序排列
            sort_feature.sort()
        else:    #特征按降序排列
            sort_feature.sort(reverse=True)
        for j in range(m):
            value = array[j,i]
            pos = sort_feature.index(value)    #相同的值，位置一样
            array[j,i] = pos
    return list(array)

def process_feature1(data):    # 每条路径的属性在AS PAIR中所有路径中的排名
    array = np.array(data)
    m =  array.shape[0]    #行数
    n = array.shape[1]    #列数
    for i in range(n-3):    #不用规则化的列数
        feature = list(array[:,i])
        sort_feature = list(set(feature))
        sort_feature.sort()
        for j in range(m):
            value = array[j,i]
            pos = sort_feature.index(value)
            array[j,i] = pos
    return list(array)

def process_feature2(data):    # 每条路径的属性在AS PAIR中与均值比大小
    array = np.array(data)
    m =  array.shape[0]    #行数
    n = array.shape[1]    #列数
    for i in range(n-2):
        feature = list(array[:,i])
        s = 0
        for l in feature:
            s += int(l)
        feature_mean = s/len(feature)
        for j in range(m):
            value = int(array[j,i])
            if value < feature_mean:
                array[j,i] = 0
            else:
                array[j,i] = 1
    return list(array)

def process_feature3(data):    # 每条路径的属性在AS PAIR中是最大，最小，或者都不是 3最大，2中间，1最小
    array = np.array(data)
    m =  array.shape[0]    #行数
    n = array.shape[1]    #列数
    for i in range(n-2):
        feature = [int(x) for x in array[:,i]]
        feature_max = max(feature)
        feature_min = min(feature)
        for j in range(m):
            value = int(array[j,i])
            if value == feature_max:
                array[j,i] = 3
            elif value == feature_min:
                array[j,i] = 1
            else:
                array[j,i] = 2
    return list(array)


# types: Content-0|Enterprise-1|Transit/Access-2
def as2types_dict():
    as2typesp_file = files_path + 'data/aspath/20190901.as2types.txt'
    df = pd.read_csv(as2typesp_file,names=['as2types'])
    as2types = df['as2types'].tolist()
    d ={}
    for r in as2types:
        astype = r.split('|')[2]
        asn = r.split('|')[0]
        if astype == 'Content':
            int_astype = 0
        elif astype == 'Enterprise':
            int_astype = 1
        else:
            int_astype = 2
        d[asn] = int_astype

    return d                

def relationship_dict():     
    as_relationship_file = files_path + 'data/aspath/20190901.as-rel.txt'
    df = pd.read_csv(as_relationship_file,names=['rel'])
    rel = df['rel'].tolist()
    d ={}
    for r in rel:
        as1 = r.split('|')[0]
        as2 = r.split('|')[1]
        relationship = int(r.split('|')[2])
        try:
            tmp = d[as1]
            tmp[as2] = relationship
            d[as1] = tmp
        except:
            d1 = {}
            d1[as2] = relationship
            d[as1] = d1

    return d                
             
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


def count_by_tpath_length(ipath,tpath,analyze_dict):    #按真实路径长度统计推测路径的长度分布
    tlen = len(tpath)-2     #length without prefix
    ilen = len(ipath)-2     #length without prefix
    try:
        tmp = analyze_dict[tlen]    #是一个List,长度为4[shorter_num,same_num,longer_num,exact_same_num]
    except:
        tmp = [0,0,0,0]
    if ilen > tlen:
        num = tmp[2]
        num += 1
        tmp[2] = num
    elif ilen == tlen:
        num = tmp[1]
        num += 1
        tmp[1] = num
        if ipath == tpath:
            num = tmp[3]
            num += 1
            tmp[3] = num
    else:
        num = tmp[0]
        num += 1
        tmp[0] = num
    analyze_dict[tlen] = tmp

#jaccard相似性
def jaccard_index(li1,li2):
    intersection = []
    union = li2.copy()
    for l in li1:
        if l in li2:
            intersection.append(l)
        else:
            union.append(l)
    similarity = len(intersection)/len(union)
    return similarity

def geo_relation(as_rank_info,cc_dict,triplet):

    src_as = triplet[0]
    stitch_as = triplet[1]
    dst_as = triplet[2]
    country_src = as_rank_info[src_as]['country']
    country_stitch = as_rank_info[stitch_as]['country']
    country_dst = as_rank_info[dst_as]['country']
    continent_src = cc_dict[country_src]
    continent_stitch = cc_dict[country_stitch]
    continent_dst = cc_dict[country_dst]
    if country_src == country_dst:
        if country_stitch == country_src:
            rel = 1
        else:
            if continent_stitch == continent_src:
                rel = 2
            else:
                rel = 3
    else:
        if continent_src == continent_dst:
            if continent_stitch == continent_src:
                rel = 4
            else:
                rel = 5
        else:
            if continent_stitch == continent_src or continent_stitch == continent_dst:
                rel = 6
            else:
                rel = 7
    return rel



def analyze_stitching_path(known_aspath,left_split_path, prefix2as, validate_aspath, multi_degree, as_rank_info, frequency_dict, rel_dict, as2type,
                           two_seg_key, joint_dir, current_dir):
    country2continent = files_path + 'data/aspath/country2continent.json'
    fp = open(country2continent)
    str_js = ujson.load(fp)  # 从文本中读取，str
    cc_dict = ujson.loads(str_js)  # 国家对应的大洲
    fp.close()

    unprocessed = []

    all_path = []
    single_shortest = []
    multi_shortest = []  # 有多条最短路径
    snum = 0
    mnum = 0
    nopath = 0
    single_correct = 0  # 只有一条最短推测路径时第一条为目标路径的数量
    multi_correct = 0  # 有多条最短推测路径时第一条为目标路径的数量
    single_jaccard = []
    multi_jaccard = []
    jaccard = []
    single_shortest_analyze_dict = {}  # 只有一条最短推测路径：按真实路径长度统计推测路径的分布
    multi_shortest_analyze_dict = {}  # 有多条最短推测路径：按真实路径长度统计推测路径的分布
    single_jaccard_by_length = {}
    multi_jaccard_by_length = {}
    if not os.path.exists(joint_dir):
        os.makedirs(joint_dir)
    single_shortest_file = joint_dir + 'single_shortest.csv'
    multi_shortest_file = joint_dir + 'multi_shortest.csv'
    all_file = joint_dir + 'all.csv'
    unprocessed_file = joint_dir + 'unprocessed.csv'
    if os.path.exists(single_shortest_file):
        os.remove(single_shortest_file)
    if os.path.exists(multi_shortest_file):
        os.remove(multi_shortest_file)
    if os.path.exists(all_file):
        os.remove(all_file)
    if os.path.exists(unprocessed_file):
        os.remove(unprocessed_file)
    for row in two_seg_key:
        prefix_pair = row[0]
        infers = row[1]
        vpath = validate_aspath[prefix_pair]
        prefixs = prefix_pair.split('-')
        vpath = prefixs[0] + '->' + vpath + '->' + prefixs[1]
        tmp = []
        old_method = []
        for infer in infers:
            start_prefix = infer[0]
            joint_point = infer[1]
            end_prefix = infer[2]
            left_status = False
            try:
                start_as = prefix2as[start_prefix]
                key1 = start_as + '-' + joint_point
                seg1 = left_split_path[key1]
                left_status = True
            except:
                key1 = joint_point + '-' + start_prefix
                try:
                    seg1 = known_aspath[key1]
                except:
                    jaccard.append(0)
                    continue
                    
            key2 = joint_point + '-' + end_prefix
            try:
                seg2 = known_aspath[key2]
            except:
                jaccard.append(0)
                continue
            for s1 in seg1:
                if left_status:
                    left = s1.split('->')
                else:
                    left = s1.split('->')[0:-1]
                    left.reverse()
                for s2 in seg2:
                    right = s2.split('->')[1:-1]
                    ipath = left + right
                    for asn in ipath:  # loop-free
                        if ipath.count(asn) > 1:
                            continue
                    valley_free = regular.valley_free(ipath, rel_dict)
                    if valley_free == False:
                        continue
                    try:
                        left_frequency = frequency_dict[s1]
                        right_frequency = frequency_dict[s2]
                        frequency = max(left_frequency, right_frequency)
                    except:
                        frequency=1
                    length = len(ipath)

                    # degree = neighbor_num_dict[int(joint_point)]
                    try:
                        stitch_global_degree = as_rank_info[joint_point]['degree']['globals']
                    except:
                        stitch_global_degree = 0
                    mdegree = multi_degree[joint_point]
                    # tdegree = transit_degree[int(joint_point)]
                    try:
                        tdegree = as_rank_info[joint_point]['degree']['transits']
                    except:
                        tdegree = 0
                    # second_as = int(ipath[1])
                    # second_degree = neighbor_num_dict[int(second_as)]
                    second_as = ipath[1]
                    try:
                        sec_transit_degree = as_rank_info[second_as]['degree']['transits']
                    except:
                        sec_transit_degree = 0
                    try:
                        sec_global_degree = as_rank_info[second_as]['degree']['globals']
                    except:
                        sec_global_degree = 0
                    try:
                        astype = as2type[joint_point]
                    except:
                        astype = -1
                    # joint_num = len(infers)
                    triplet = [ipath[0], joint_point, ipath[-1]]
                    try:
                        geo_relationship = geo_relation(as_rank_info, cc_dict, triplet)
                    except:
                        geo_relationship = 8
                    ipath.insert(0, start_prefix)
                    ipath.append(end_prefix)
                    tmp.append([length, frequency, tdegree, mdegree, sec_transit_degree, sec_global_degree,
                                stitch_global_degree, astype, geo_relationship, '->'.join(ipath)])
                    old_method.append([length, '->'.join(ipath)])
        tmp.sort()
        old_method.sort()
        if len(tmp) == 0:
            nopath += 1
            continue

        joint_num = len(tmp)
        for i in range(joint_num):
            tmp[i].insert(-1, joint_num)

        for t in tmp:
            t.append(vpath)
            unprocessed.append(t)
        if len(unprocessed) >= 100000:
            df = pd.DataFrame(unprocessed)
            df.to_csv(unprocessed_file, index=False, header=None, mode='a')
            unprocessed = []

        data = process_feature(tmp)

        for d in data:
            row = list(d)
            row.append(vpath)
            all_path.append(row)

        if len(all_path) >= 100000:
            df = pd.DataFrame(all_path)
            df.to_csv(all_file, index=False, header=None, mode='a')
            all_path = []

        length = []
        for row in data:
            length.append(row[0])
        shortest = min(length)
        if length.count(shortest) > 1:
            mnum += 1
            infer_path = old_method[0][-1]
            if infer_path == vpath:
                multi_correct += 1
            similarity = jaccard_index(infer_path.split('->'), vpath.split('->'))
            count_by_tpath_length(infer_path.split('->'), vpath.split('->'), multi_shortest_analyze_dict)
            multi_jaccard.append(similarity)
            vlen = len(vpath.split('->')) - 2
            try:
                tmp = multi_jaccard_by_length[vlen]
                tmp.append(similarity)
                multi_jaccard_by_length[vlen] = tmp
            except:
                multi_jaccard_by_length[vlen] = [similarity]
            for d in data:
                # if d[0] == shortest:
                row = list(d)
                row.append(vpath)
                multi_shortest.append(row)
        else:
            snum += 1
            # infer_path = list(data[0])[-1]
            infer_path = old_method[0][-1]
            if infer_path == vpath:
                single_correct += 1
                
            similarity = jaccard_index(infer_path.split('->'), vpath.split('->'))
            count_by_tpath_length(infer_path.split('->'), vpath.split('->'), single_shortest_analyze_dict)
            single_jaccard.append(similarity)
            vlen = len(vpath.split('->')) - 2
            try:
                tmp = single_jaccard_by_length[vlen]
                tmp.append(similarity)
                single_jaccard_by_length[vlen] = tmp
            except:
                single_jaccard_by_length[vlen] = [similarity]
            for d in data:
                row = list(d)
                row.append(vpath)
                single_shortest.append(row)
        if len(single_shortest) >= 100000:
            df_single_shortest = pd.DataFrame(single_shortest)
            df_single_shortest.to_csv(single_shortest_file, index=False, header=None, mode='a')
            single_shortest = []
        if len(multi_shortest) >= 100000:
            df_multi_shortest = pd.DataFrame(multi_shortest)
            df_multi_shortest.to_csv(multi_shortest_file, index=False, header=None, mode='a')
            multi_shortest = []

    print('joint_path_num:', len(two_seg_key))
    print('nopath:', nopath)
    print('single_shortest num:', snum)
    print('multi_shortest num:', mnum)
    if snum == 0:
        print('single_shortest correctness:', 0)
    else:
        print('single_shortest correctness:', single_correct / snum)
    if mnum == 0:
        print('multi_shortest correctness:', 0)
    else:
        print('multi_shortest correctness:', multi_correct / mnum)
    print('total correctness:', (single_correct + multi_correct) / (snum + mnum))
    jaccard = single_jaccard + multi_jaccard
    print('single jaccard:', np.mean(np.array(single_jaccard)))
    print('multi jaccard:', np.mean(np.array(multi_jaccard)))
    print('total jaccard:', np.mean(np.array(jaccard)))

    result_file = current_dir + '/result.txt'
    fp = open(result_file, 'a')
    fp.write('joint_path_num:%d\n' % len(two_seg_key))
    fp.write('nopath:%d\n' % nopath)
    fp.write('single_shortest num:%d\n' % snum)
    fp.write('multi_shortest num:%d\n' % mnum)
    #fp.write('single_shortest correctness:%f\n' % (single_correct / snum))
    fp.write('multi_shortest correctness:%f\n' % (multi_correct / mnum))
    fp.write('total correctness:%f\n' % ((single_correct + multi_correct) / (snum + mnum)))
    fp.write('single jaccard:%f\n' % np.mean(np.array(single_jaccard)))
    fp.write('multi jaccard:%f\n' % np.mean(np.array(multi_jaccard)))
    fp.write('total jaccard:%f\n' % np.mean(np.array(jaccard)))
    fp.close()

    df_single_shortest = pd.DataFrame(single_shortest)
    df_single_shortest.to_csv(single_shortest_file, index=False, header=None, mode='a')

    df_multi_shortest = pd.DataFrame(multi_shortest)
    df_multi_shortest.to_csv(multi_shortest_file, index=False, header=None, mode='a')

    df = pd.DataFrame(all_path)
    df.to_csv(all_file, index=False, header=None, mode='a')
    mean_single_jaccard_by_length = {}
    for key in single_jaccard_by_length.keys():
        item = single_jaccard_by_length[key]
        mean_single_jaccard_by_length[key] = np.mean(np.array(item))
    mean_multi_jaccard_by_length = {}
    for key in multi_jaccard_by_length.keys():
        item = multi_jaccard_by_length[key]
        mean_multi_jaccard_by_length[key] = np.mean(np.array(item))

    return single_shortest_analyze_dict, multi_shortest_analyze_dict, mean_single_jaccard_by_length, mean_multi_jaccard_by_length

def country2continent(as_rank_info):
    country = []
    for key in as_rank_info.keys():
        name = as_rank_info[key]['country']
        if len(name) > 2:
            continue
        if name not in country:
            country.append(name)
    country2continent = files_path + 'data/aspath/country_continent.csv'
    df = pd.read_csv(country2continent,names=['1','2','continent','country'])
    cc_dict = {}
    cc_dict['NA'] = 'Africa'
    for row in df[['continent','country']].values.tolist():
        key = row[1]
        item = row[0]
        cc_dict[key] = item
    for c in country:
        try:
            cc_dict[c]
        except:
            print(c)
    cc_json = files_path + 'data/aspath/country2continent.json'
    fp = open(cc_json, 'w')
    json_str = ujson.dumps(cc_dict)
    ujson.dump(json_str, fp)
    fp.close()        

def multi_process(function,argc,num_thread):
    # 多线程下载
    p = Pool(num_thread)
    p.map(function, argc)
    p.close()
    p.join()

def main(test_ratio):
    root_dir = files_path + 'data/aspath/path_by_vantage/'
    vantage_aspath_list = os.listdir(root_dir)
    para = []
    for file in vantage_aspath_list:
        para.append([file,test_ratio])

    multi_process(child,para,num_thread=1)
    '''
    for file in vantage_aspath_list:
        current_dir = root_dir + file
        print(file)
        child(test_ratio,current_dir)
    '''
def child(para):
    file = para[0]
    if file == '.DS_Store':
        return
    print(file)
    test_ratio = para[1]
    root_dir = files_path + 'data/aspath/path_by_vantage/'
    current_dir = root_dir + file

    print('multi_graph')
     #求每条路径的频率
    frequency_file = current_dir + '/var/frequency.txt'
    f = open(frequency_file,'rb')
    frequency_dict = pickle.load(f)
    f.close()
    frequency_file = current_dir + '/var/left_frequency.txt'
    f = open(frequency_file,'rb')
    left_frequency_dict = pickle.load(f)
    f.close()
    frequency_dict.update(left_frequency_dict)
    print('get frequency successfully')
    
    multi_degree,G,known_aspath,left_split_path = get_basic_info(current_dir)
    print('get basic info successfully')

    as_rank_info_file = files_path + 'data/aspath/as_rank_info.json'
    fp = open(as_rank_info_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    as_rank_info = ujson.loads(str_js)    #PREFIX属于的AS
    fp.close()
    print('get asrank successfully!')
    
    prefix2as_file = files_path + 'data/aspath/prefix2as.json'
    fp = open(prefix2as_file)
    str_js = ujson.load(fp)    #从文本中读取，str
    prefix2as = ujson.loads(str_js)    #PREFIX属于的AS
    fp.close()
    print('get prefix2as successfully!')
    
    rel_dict = relationship_dict()
    print('get AS relationship successfully!')
    
    as2type = as2types_dict()
    print('get AS type successfully!')
    
    result_file =  current_dir + '/result.txt'
    validate_dir =  current_dir + '/validate'
    validate_aspath_list = os.listdir(validate_dir)
    
    for row in validate_aspath_list:
        child_dir_name = row

        if child_dir_name == '.DS_Store':
            continue
        
        child_dir = validate_dir + '/' + child_dir_name + '/'
        validate_file = child_dir + 'ground_path.txt'
        cn_file = child_dir + 'common_neighbors.txt'
        joint_dir = child_dir + 'joint_path/'
        print(child_dir_name )
        f = open(validate_file,'rb')
        validate_aspath = pickle.load(f)
        f.close()
        
        print('get validate data successfully')

        fp = open(result_file,'a')
        fp.write(row)
        fp.write('\n')
        fp.close()
        has_path,no_path_key,one_seg_key,two_seg_key,other_seg_key = shortest_path(G,validate_aspath,cn_file,current_dir)
        single_shortest_analyze_dict, multi_shortest_analyze_dict, mean_single_jaccard_by_length, mean_multi_jaccard_by_length = analyze_stitching_path(
            known_aspath, left_split_path, prefix2as, validate_aspath, multi_degree, as_rank_info, frequency_dict, rel_dict, as2type, two_seg_key,
            joint_dir,current_dir)

        index_list = ['shorter','same','longer','exact_same']
        result_dir = current_dir + '/result/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = result_dir + child_dir_name
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        single_shortest_result_file = result_dir + '/baseline_single_result_%s.csv' % (test_ratio)
        df_single = pd.DataFrame(single_shortest_analyze_dict, index=index_list)
        df_single.to_csv(single_shortest_result_file)
        df_single = pd.DataFrame(mean_single_jaccard_by_length, index=['jaccard'])
        df_single.to_csv(single_shortest_result_file, mode='a')

        multi_shortest_result_file = result_dir + '/baseline_multi_result_%s.csv' % (test_ratio)
        df_multi = pd.DataFrame(multi_shortest_analyze_dict, index=index_list)
        df_multi.to_csv(multi_shortest_result_file)
        df_multi = pd.DataFrame(mean_multi_jaccard_by_length, index=['jaccard'])
        df_multi.to_csv(multi_shortest_result_file, mode='a')
        print('mean_single_jaccard_by_length')
        print(mean_single_jaccard_by_length)
        print('single_shortest_analyze_dict')
        print(single_shortest_analyze_dict)
        print('mean_multi_jaccard_by_length')
        print(mean_multi_jaccard_by_length)
        print('multi_shortest_analyze_dict')
        print(multi_shortest_analyze_dict)
    
# if __name__=="__main__":
#     print('ssss')






    







