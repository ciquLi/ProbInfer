#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:04:24 2019

@author: lixionglve
"""



def relationship_dict():     
    as_relationship_file = files_path + 'data/aspath/20190101.as-rel.txt'
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

def get_as_relationship(relationship,as1,as2):
    #<provider-as>|<customer-as>|-1 反过来为1 peer peer 为0  ,sibling - sibling 2
    try:
        rel = relationship[as1][as2]
    except:
        try:
            rel = -relationship[as2][as1]
        except:
            rel = 2
    
    return rel

############
'''
判断AS路径是否符合valley-free原则
准则一:一条 AS 路径中最多只有一条 P2P 连接;
准则二:一条 AS 路径中，如果有一条 P2C 连接，它后面就不可能是 C2P 连接 ，只可能是 P2C 或 S2S 连接;
准则三:P2C 连接后不可能是 P2P 连接;
准则四:P2P 连接后不可能是 C2P 连接，只可能是 P2C 或 S2S 连接。

as_list:  [as1,as2,as3,as4]
'''
############
def valley_free(as_list,rel_dict):
    res = True
    relationship = []
    for i in range(len(as_list)-1):
        rel = get_as_relationship(rel_dict,as_list[i],as_list[i+1])
        relationship.append(rel)
    if relationship.count(0) > 1:
        res = False
        return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 1 in relationship[pos+1:]:
            res = False
            return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 0 in relationship[pos+1:]:
            res = False
            return res
    if 0 in relationship:
        pos = relationship.index(0)
        if 1 in relationship[pos+1:]:
            res = False
            return False
    return res
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    