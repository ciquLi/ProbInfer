#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:36:41 2019

@author: lixionglve
"""
import data_preprocessing
import get_ground_path
import get_validate_data
import multigraph_infer
#import predict



if __name__=="__main__":
    #原始数据处理

    #data_preprocessing.main()

    test_ratios = [0.9]
    for test_ratio in test_ratios:
        #get_ground_path.main(test_ratio)
        #get_validate_data.main()
        print('test ratio: %s'%(test_ratio))
        multigraph_infer.main(test_ratio)
        #predict.main(test_ratio)
