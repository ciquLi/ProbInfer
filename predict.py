#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:25:11 2019

@author: lixionglve
"""
import pickle
import time
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris
# from skrules import SkopeRules

current_path = os.getcwd()  #返回当前文件所在的目录   
parent_path = os.path.dirname(current_path) #获得current_path所在的目录,即父级目录 
files_path = parent_path + '/files/'    #获得存放数据的根目录


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

def get_similarity(file):
    res = []
    df = pd.read_csv(file,header=None)
    for v in df.values.tolist():
        text1 = v[10]
        text2 = v[11]
        #similarity = int(textdistance.hamming.normalized_similarity(text1, text2))
        #similarity = jaccard_index(text1.split('->'),text2.split('->'))
        if text1 == text2:
            similarity = 1
        else:
            similarity = 0
        row = v[0:10] + [similarity]
        row.append(text1)
        row.append(text2)
        res.append(row)
    df_res = pd.DataFrame(res)
    df_res.columns=['length','frequency',
                     'stitch_transit_degree','stitch_multidegree','sec_transit_degree','sec_global_degree',
                     'stitch_global_degree','AStype','geo_relationship','size_canpaths',
                     'label','infer_path','true_path']
    return df_res
    #similarity_file = files_path + 'data/aspath/joint_path/data_similarity.csv'
    #df_res.to_csv(similarity_file,index=False,header=None)

def train_model(train_file,name,features,current_dir):
    df_train = get_similarity(train_file)

    x_train = df_train[features]
    y_train = df_train['label']
    
    feature_names = x_train.columns
    feature_names = np.array(feature_names)
    # clf = SkopeRules(max_depth_duplication=2,n_estimators=30,precision_min=0.3,recall_min=0.1,feature_names=feature_names)
    # clf = clf.fit(x_train, y_train)
    # model_path = current_dir + '/validate/train/' + 'rulefit_' + name + '.m'
    # joblib.dump(clf, model_path)   # 保存模型
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    model_path = current_dir + '/validate/train/' + 'DecisionTree_' + name + '.m'
    joblib.dump(clf, model_path)   # 保存模型


    
def decision_tree(df_test,name,features,current_dir,mark):
    
    x_test = df_test[features]
    y_test = df_test['label']
    
    model_path = current_dir + '/validate/train/' + 'DecisionTree_' + name + '.m'
    clf = joblib.load( model_path)   # 读取模型
    
    pred = clf.predict(x_test)
    #prob = clf.predict_proba(x_test)
    #print('test score:',clf.score(x_test,y_test))
   # print('accuracy score',accuracy_score(y_test,pred))
    #df_test['pred'] = prob[:,1]
    df_test['pred'] = pred
    
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 1, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#1bb2d9')
    feature_names = x_test.columns
    feature_names = np.array(feature_names)
    print(feature_importance)
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    #plt.title('Feature Importance')
    plt.rcParams['savefig.dpi'] = 200 #图片像素
    plt.rcParams['figure.dpi'] = 200 #分辨率
    #plt.show()
    img_name = current_dir + '/result/test/' + mark + '_' + name + '.png'
    plt.savefig(img_name)
    print('feature importance finished')
    
    return df_test



def count_by_tpath_length(ipath,tpath,analyze_dict):    #按真实路径长度统计推测路径的长度分布
    tlen = len(tpath) - 2
    ilen = len(ipath) - 2
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

def analyze_jaccard(df):
    result = {}
    for row in df[['length','infer_path','true_path','pred']].values:
        length = -row[0]
        ipath = row[1]
        tpath = row[2]
        pred = row[3]
        try:
            tmp = result[tpath]
            tmp.append([pred,length,ipath])
            result[tpath] = tmp
        except:
            result[tpath] = [[pred,length,ipath]]
    res = {}
    for key in result.keys():
        tmp = result[key]
        tmp.sort(reverse = True)    #降序
        res[key] = tmp[0:1]    #预测：取pred最大的路径
    exact_same = 0
    same_length = 0
    shorter = 0
    longer = 0
    jaccard = []
    jaccard_by_length = {}
    analyze_dict = {}
    for key in res.keys():
        tpath = key.split('->')
        # for item in res[key]:
        item = res[key]
        ipath = item[2].split('->')
        similarity = jaccard_index(ipath,tpath)
        jaccard.append(similarity)
        count_by_tpath_length(ipath,tpath,analyze_dict)
        tlen = len(tpath) - 2
        try:
            tmp = jaccard_by_length[tlen]
            tmp.append(similarity)
            jaccard_by_length[tlen] = tmp
        except:
            jaccard_by_length[tlen] = [similarity]
        if len(ipath) < len(tpath):
            shorter += 1
        elif len(ipath) == len(tpath):
            same_length += 1
            if ipath == tpath:
                exact_same += 1
        else:
            longer += 1
    print('path num:',len(result))
    print('exact_same',exact_same)
    print('same_length',same_length)
    print('shorter',shorter)
    print('longer',longer)   
    print('jaccard:',np.mean(np.array(jaccard)))
    mean_jaccard_by_length = {}
    mean_jaccard_by_length_num = {}
    for key in jaccard_by_length.keys():
        item = jaccard_by_length[key]
        mean_jaccard_by_length[key] = np.mean(np.array(item))
        mean_jaccard_by_length_num[key] = [np.mean(np.array(item)),len(item)]
    print(mean_jaccard_by_length)
    return analyze_dict,mean_jaccard_by_length,mean_jaccard_by_length_num




def skope_rules(df_test,name,features,current_dir,mark):
    x_test = df_test[features]
    model_path = current_dir + '/validate/train/' + 'rulefit_' + name + '.m'
    clf = joblib.load(model_path)   # 读取模型
    pred = clf.predict(x_test)
    df_test['pred'] = pred
    rules = clf.rules_[0:3]
    for rule in rules:
        print(rule)
    return df_test



def main(test_ratio):
    print('predict')

    root_dir = files_path + 'data/aspath/path_by_vantage/'
    # vantage_aspath_list = os.listdir(root_dir)
    vantage_aspath_list = ['20211002']
    for file in vantage_aspath_list:
        if file == '.DS_Store':
            continue
        print(file)
        current_dir = root_dir + file
        features = ['length', 'frequency', 'stitch_transit_degree', 'stitch_multidegree', 'sec_transit_degree',
                    'sec_global_degree', 'stitch_global_degree', 'AStype', 'geo_relationship', 'size_canpaths']
        mark = 'importance'
        feature_analyze(test_ratio, features, current_dir,mark)
        features = ['length', 'frequency', 'stitch_transit_degree', 'stitch_multidegree', 'sec_transit_degree',
                    'sec_global_degree', 'stitch_global_degree', 'geo_relationship', 'size_canpaths']
        mark = 'predict'
        feature_analyze(test_ratio, features, current_dir, mark)

    '''
    for i in range(len(features)):
        print('removed feature %s: '%features[i])
        left_feature = features.copy()
        left_feature.remove(features[i])
        feature_analyze(test_ratio,left_feature,current_dir)
    '''

def feature_analyze(test_ratio,features,current_dir,mark):
    train_dir = current_dir + '/validate/train/joint_path/'
    train_list = os.listdir(train_dir)
    for file in train_list:
        if file == '.DS_Store':
            continue
        train_file = train_dir + file
        name = file.split('.')[0]
        train_model(train_file,name,features,current_dir)
    
    index_list = ['shorter','same','longer','exact_same']
    validate_dir = current_dir + '/validate/'
    name_list = os.listdir(validate_dir)
    for name in name_list: 
        if name == '.DS_Store' or name == 'train':
            continue
        path = validate_dir + name + '/joint_path/'
        file_list = os.listdir(path)
        file_list = ['multi_shortest.csv','single_shortest.csv']
        for file in file_list:
            if file == '.DS_Store':
                continue
            print(30*'#')
            print('analyze:',name,file)
            test_file = path + file
            df_test = get_similarity(test_file)
            test_name = file.split('.')[0]
            print('Decision tree')
            df = decision_tree(df_test,test_name,features,current_dir,mark)
            result_by_length,mean_jaccard_by_length,mean_jaccard_by_length_num = analyze_jaccard(df)
            
            result_by_length_file  = current_dir + '/result/%s'%(name) + '/%s_dt_%s_%s.csv'%(mark,test_name,test_ratio)
            df_res = pd.DataFrame(result_by_length,index=index_list)
            df_res.to_csv(result_by_length_file)
            
            df_res = pd.DataFrame(mean_jaccard_by_length,index=['jaccard'])
            df_res.to_csv(result_by_length_file,mode='a')
            print(result_by_length)

            df_res = pd.DataFrame(mean_jaccard_by_length_num,index=['jaccard','num'])
            df_res.to_csv(result_by_length_file,mode='a')
            print(result_by_length)
            
            # print(30*'-')
            # print('rulefit %s'%test_name)
            # df = skope_rules(df_test,test_name,features,current_dir,mark)
            # result_by_length,mean_jaccard_by_length = analyze_jaccard(df)
            # result_by_length_file  = current_dir + '/result/test' + '/%s_rulefit_%s_%s.csv'%(mark,test_name,test_ratio)
            # df_res = pd.DataFrame(result_by_length,index=index_list)
            # df_res.to_csv(result_by_length_file)
            
            # df_res = pd.DataFrame(mean_jaccard_by_length,index=['jaccard'])
            # df_res.to_csv(result_by_length_file,mode='a')
            # print(result_by_length)
            

if __name__=="__main__":
    test_ratio = 0.3
    features = ['length','frequency','tdegree','joint_mdegree','second_degree','joint_degree','as2type','joint_num','geo_relationship']
    feature_analyze(test_ratio,features)
    '''
    for i in range(len(features)):
        print('removed feature %s: '%features[i])
        left_feature = features.copy()
        left_feature.remove(features[i])
        feature_analyze(test_ratio,left_feature)
    '''
            
        
    
    






