#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 13:43:24 2018

@author: lixionglve
"""

from bs4 import BeautifulSoup
import requests, sys
import time,datetime
import os
from pget import Downloader
import multiprocessing
from multiprocessing.dummy import Pool
import subprocess




current_path = os.getcwd()  #返回当前文件所在的目录
parent_path = os.path.dirname(current_path) #获得current_path所在的目录,即父级目录
files_path = parent_path + '/files/'    #获得存放数据的根目录

def is_valid_date(date):
  '''判断是否是一个有效的日期字符串'''
  try:
    time.strptime(date, "%Y-%m-%d")
    return True
  except:
    return False
# 获取html内容
def getHtml(url):
    res = requests.get(url)
    html = BeautifulSoup(res.text,features="lxml")
    return html
def download_file(param):
        # 下载文件
    filename, save_directory, url = param
    
    file_path = save_directory + '/' + filename
    
    if not os.path.exists(file_path):  # To prevent download exists files
        if not os.path.exists(save_directory) :  # To create directory
            os.makedirs(save_directory)
            
        try:
            r = requests.get(url, stream=True)
        except requests.exceptions.ReadTimeout as e:
            print('Error-----------文件下载失败,服务器长时间无响应: ', filename)
        except requests.exceptions.ConnectionError as e:
            print('Error-----------文件下载失败,服务器长时间无响应: ', filename)
        size_mb = int(r.headers.get('Content-Length')) / (1024 ** 2)
        try:
            print('Start download %s'%filename,'%.2f'%size_mb)
            t1 = time.time()
            '''
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
            '''
            #cmd='wget -P %s %s' % (save_directory,url)
            #subprocess.call(cmd,shell=True)
         
            
            state = True
            while state:
                downloader = Downloader(url, file_path, 8)
                downloader.start_sync()
             #   cmd='wget -P %s %s' % (save_directory,url)
              #  print(save_directory)
               # subprocess.call(cmd,shell=True)
                fsize = os.path.getsize(file_path)
                fsize_mb = int(fsize) / (1024 ** 2)
                if int(fsize_mb) == int(size_mb):
                    state = False
                else:
                    print('%s download again'%filename)
            time_used = time.time() - t1
            print('%s Download success'%filename)
            print('%s time used'%time_used)
        except UnicodeEncodeError:
            print('%s Download EncodeError'%filename)
            
def start_download(download):
        # 多线程下载
        p = Pool(3)
        p.map(download_file, download)
        p.close()
        p.join()   
        

    

def down_bgp(n=1):   #下载从今天开始往前n天的数据，默认为1
    rrc = ['rrc00','rrc01','rrc03','rrc04','rrc05','rrc06','rrc07'
           ,'rrc10','rrc11','rrc12','rrc13','rrc14','rrc15','rrc16'
           ,'rrc18','rrc19','rrc20','rrc21','rrc22','rrc23']    #可用站点   2、8、9、17站点不可用
    today = datetime.datetime.today()
    bgp_url = []    #存储所有下载地址
    for delta in range(n):
        date = today - datetime.timedelta(days=delta)
        str_date =  datetime.date.strftime(date, "%Y%m%d")
        year = str_date[0:4]
        month = str_date[4:6]
        for collector in rrc:
            save_dir = files_path + 'data/raw_data/bgp/' + str_date
            if not os.path.exists(save_dir) :  # 如果目录不存在，则创建目录
                os.mkdir(save_dir)
            date_url = 'http://data.ris.ripe.net/' + collector + '/' + year + '.' + month + '/'
            html = getHtml(date_url)
            for a in (html.find_all('a')):
                href = a.get('href')
                if href.find('bview') != -1 and href.find(str_date) != -1:
                    #url = 'http://data.ris.ripe.net/' + collector + '/' + year + '.' + month + '/bview.' + str_date + '.0000.gz'
                    filename = collector + '-' + href[6:]
                    url = date_url + href
                    print(url,filename)
                    bgp_url.append([filename,save_dir,url])
    return bgp_url


if __name__=="__main__":
    #多线程下载
    lock = multiprocessing.Lock()

    #下载bgp数据
    days = 1    #从当前日期开始，往前要下载的天数
    bgp_url = down_bgp(days)
    start_download(bgp_url)
   
    
        
    











