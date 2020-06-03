#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:57:01 2019

@author: lixionglve
"""

import ujson
import os
from bs4 import BeautifulSoup
import requests, sys
from multiprocessing import Pool
        
current_path = os.getcwd()  #返回当前文件所在的目录   
parent_path = os.path.dirname(current_path) #获得current_path所在的目录,即父级目录 
files_path = parent_path + '/files/'    #获得存放数据的根目录

def multi_process(function,argc,num_process):
    # 多进程
    p = Pool(num_process)
    result = p.map(function, argc)
    p.close()
    p.join()
    return result

    # 获取html内容
def getHtml(url):
    res = requests.get(url,timeout=30)
    html = BeautifulSoup(res.text,'html.parser')
    return html

def get_asn():
    asn_json = files_path + 'data/aspath/test/all_asn.json'
    fp = open(asn_json)
    json_str = ujson.load(fp)
    asn_dict = ujson.loads(json_str)
    fp.close()
    return list(asn_dict.keys())

def get_data_from_web(asn):
    url = 'http://as-rank.caida.org/asns/' + asn
    html = getHtml(url)
    script = html.find_all('script')
    for s in script:
        t = s.get('type')
        if t == 'application/ld+json':
            string = s.string
            js_data = ujson.loads(string)
            return js_data

def caculate(asn_list):
    asn_info = {}
    get = 0
    fail = 0
    pid = os.getpid()
    for asn in asn_list:
        try:
            js_data = get_data_from_web(asn)
            get += 1
        except:
            fail += 1
            continue
        asn_info[asn] = js_data
        if get%10 == 0 :
            print('pid',pid)
            print('get:',get)
            print('total',get+fail)
        if fail%10 == 0 and fail > 0:
            print('pid',pid)
            print('fail',fail)
            print('total',get+fail)
    return asn_info
    
if __name__=="__main__":
    as_rank_json = files_path + 'data/aspath/as_rank_info.json'
    
    n = 10    #线程数量
    total_asn_list = get_asn()    #所有ASN
    asn_list = []    #没有得到内容的ASN
    asn_info = {}
    if os.path.exists(as_rank_json):
        fp = open(as_rank_json)
        str_js = ujson.load(fp)
        asn_info = ujson.loads(str_js)
        for asn in total_asn_list:
            try:
                asn_info[asn]
            except:
                asn_list.append(asn)
        
    length = len(asn_list)
    para = []
    seg = length//n
    for i in range(n - 1):
        tmp = asn_list[i * seg:(i + 1) * seg]
        para.append(tmp)
    tmp = asn_list[(n - 1) * seg:]
    para.append(tmp)

                
    
    #while len(asn_info) < length: 
    result = multi_process(caculate,para,n)
    for r in result:
        asn_info.update(r)

    fp = open(as_rank_json, 'w')
    json_str = ujson.dumps(asn_info)
    ujson.dump(json_str, fp)
    fp.close()
    
