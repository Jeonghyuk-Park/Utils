import glob
import json
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import openslide
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, resize, downscale_local_mean
import pickle
import random
import time
import os

"""
현재 계획
각 클래스의 리스트를 인풋으로 받으면
적어도 0이상의 region을 가지고있는 file list를 dataset ind list (0, 1, 01)에 따라 반환
이를 path로 변환하여 랜덤 샘플링해주는 함수를 포함

HOW TO USE

import Data_feeder_dx

dfd = Data_feeder_dx.Load(['LIHC'], ['0','1'], ['positive', 'negative'], ['Hepatocellular carcinoma, NOS'])

lists = dfd.list_gen_fn()

"""
def open_rgb(img_path, level): 
    try:
        slide_img = openslide.OpenSlide(img_path)
        rgb_img = np.array(slide_img.read_region([0, 0],
                                                 level, 
                                                 slide_img.level_dimensions[level]))[:,:,:3]
        return rgb_img
    except:
        print('cannot open', img_path)
        return False
    
class Load:
    """
    svs list 를 생성하고, annotation image 로부터 acceess coordinate 를 np.where로 생성
    패치를 생성하는 하위 함수를 생성
    """
    
    def __init__(self, 
                 cohort_list = [],
                 dataset_ind_list = [], 
                 class_list = [], 
                 subtype_list = []):
        """
        cohort_list = ['STAD','COAD'...]
        class_list = ['positive', 'negative']
        dataset_ind_list = [0], [1], [0,1]... 
        주의 : 위 3개는 무조건 입력되어야함.
        subtype_list = ['Adenocarcinoma, NOS', ...]]
        섭타입 리스트는 비어있을 때 별도로 필터링하지 않음
        """
        
        self.cohort_list = cohort_list
        self.dataset_ind_list = dataset_ind_list
        self.class_list = class_list
        self.subtype_list = subtype_list
        
        self.df = pd.read_csv("/nfs/ywkim/GreenCross/GitHub/NIHdataset_DX/TCGA_dataset_0531_2fold.csv", index_col = [0])
        #데이터셋 정리된 csv 를 pandas 로 read
        
        assert len(self.cohort_list) > 0
        assert len(self.class_list) > 0
        assert len(self.dataset_ind_list) > 0
        #리스트로서 기본 인풋들이 들어왔는지 체크
        
        if len(self.subtype_list) == 0:
            self.df_called = self.df.query('cohort == @self.cohort_list & dataset_ind == @self.dataset_ind_list')
            print("cohort : ", self.cohort_list,
                  "\ndataset ind : ", self.dataset_ind_list,
                  "\nNon-subtype")
        else:
            self.df_called = self.df.query('cohort == @self.cohort_list & dataset_ind == @self.dataset_ind_list & subtype == @self.subtype_list')
            print("cohort : ", self.cohort_list, 
                  "\ndataset ind : ", self.dataset_ind_list,
                  "\nsubtype : ", self.subtype_list)
        #subtype을 별도 지정하지 않은 경우 모든 subtype 에 대해서 진행
        
        
        #class_list (음성 양성) 중 하나만 출력할 것인지, 둘다 출력할 것인지 선택, annotation region > 0 이상만 각각 클래스 리스트로
        
        self.svs_list = {}
        self.access_coord = {}
        self.raw_path = {}

        
        for class_pn in self.class_list:
            
            self.svs_list[class_pn] = list((self.df_called.query('{}_area > 0'.format(class_pn[:3])).index))
            #센터 번호가 알파벳 두자리인데 ,앞 한자리만을 떼어와서 체크섬으로 이용
            #svs 
            self.access_coord[class_pn] = {}
            
            for svs_name in tqdm(self.svs_list[class_pn]):
                svs_class = self.df_called.loc[svs_name]['cohort']
                svs_base_path = '/ssd5/NIH/TCGA_DX/{}/processed/patch/positive/'.format(svs_class)
                raw_base_path =  '/ssd5/NIH/TCGA_DX/{}/raw/positive/'.format(svs_class)
                mask = imread(svs_base_path + svs_name + '/0000_{}_tissue_mask.png'.format(class_pn))
                self.access_coord[class_pn][svs_name] = np.where(mask > 128)
                self.raw_path[svs_name] = raw_base_path + svs_name
    
            print(class_pn, 'Memory Loaded')
            
            
        print("\nclass load_no checksum")            
        for class_pn in self.class_list:
            checksum_list = [x[5] for x in self.svs_list[class_pn]]
            print(class_pn, len(self.svs_list[class_pn]), ''.join(checksum_list))            
            
    def debug(self):
        return self.svs_list, self.access_coord, self.raw_path


    def patch_gen_fn(self,
                     svs_name, 
                     label,
                     isAscii = True):
        """
        svs_name을 받았을 때, 파일명을 키로 좌표 리스트를 받아서, 랜덤한 좌표를 선택하고, 패치를 생성하는 함수
        """
        
        level = 1
        patch_size = 299
        raw_path = self.raw_path

        if isAscii == True:
            svs_name = svs_name.decode('ascii')
            #file_path = file_path.decode('ascii') #file path 가 binary로 들어오므로 decoding 필요, dataset mapping 전용
        file_path = raw_path[svs_name]

        access_coord = self.access_coord


        class_pn = {1:'positive', 0:'negative'}
        file_coord_list = access_coord[class_pn[label[0]]][svs_name] # coord_mem dict에서, 좌표 list를 반환
        randint = np.random.randint(len(file_coord_list[0])) # 좌표 리스트 랜덤 indice 선택
        i,j = file_coord_list[0][randint], file_coord_list[1][randint]

    
        def open_patch(img_path, 
                       coords, 
                       level=level, 
                       patch_size=patch_size, 
                       coord_resized_factor = 4): 
            """
            좌표 구할 때 리사이즈 팩터는 4 (레벨2 마스크임)
            i, j 를 역으로 ([1],[0]) 으로 해야 동작함
            현재 이 코드는 레벨 1에서 동작하도록, 리전 영역(레벨 0 기준)에 4배를 곱하게 되어있음        
            """
            coord_i = int(coord_resized_factor*coords[1])
            coord_j = int(coord_resized_factor*coords[0])

            try:
                slide_img = openslide.OpenSlide(img_path)
                read_region_coords = [coord_i*4 - int(patch_size*4/2) + np.random.randint(coord_resized_factor*4),
                                      coord_j*4 - int(patch_size*4/2) + np.random.randint(coord_resized_factor*4)]
                #좌표를 레벨0기준에서 받기 때문에, 4배 곱하고, 랜덤값으로 리사이즈에 의한 효과를 보정
                rgb_img = np.array(slide_img.read_region(read_region_coords, level, [patch_size,patch_size]))[:,:,:3]
                return rgb_img
            except:
                print('cannot open', img_path)
                return False

        patch = open_patch(file_path, level=1, coords=[i,j], patch_size=patch_size)
        label = np.array(label, dtype=np.float32)

        return np.array(patch/255, dtype=np.float32), label        
        
    def list_gen_fn(self, shuffle_repeat = 100000):
        """
        Sampling 방식에 대하여
        1. positive는 case 베이스
            케이스만 랜덤으로 뽑아줌
        2. negative는 region 베이스
            영역 값이 샘플링될 확률임
        """
        assert len(self.class_list) == 2
        svs_list = self.svs_list
        #len(svs_list['positive'])
        
        cache = 0.00
        for ind in range(len(svs_list['negative'])):
            cache += self.df.loc[svs_list['negative'][ind]]['neg_area']
        neg_p = []
        for ind in range(len(svs_list['negative'])):
            neg_p.append(self.df.loc[svs_list['negative'][ind]]['neg_area']/cache)
             
        data_list_buffer = []
        label_list_buffer = [] 

        for path_ind in tqdm(range(shuffle_repeat)):
            pos_ind = np.random.choice(len(svs_list['positive']))
            data_list_buffer.append(svs_list['positive'][pos_ind])
            label_list_buffer.append([1,0])
            neg_ind = np.random.choice(len(svs_list['negative']), p=neg_p)
            data_list_buffer.append(svs_list['negative'][neg_ind])
            label_list_buffer.append([0,1])

        return data_list_buffer, label_list_buffer   
    
    def list_gen_fn_v2(self, shuffle_repeat = 100000):
        """
        Sampling 방식에 대하여
        1. positive는 case 베이스
            케이스만 랜덤으로 뽑아줌
        2. negative는 region 베이스
            영역 값이 샘플링될 확률임
        v2는 
        """
        if len(self.class_list) == 2:
            data_list_buffer, label_list_buffer = list_gen_fn(self)
            
        elif len(self.class_list) == 1:
            
            svs_list = self.svs_list    
            
            data_list_buffer = []
            label_list_buffer = []       
            
            if (self.class_list)[0] == 'positive':
                print('Single Positive')
                for path_ind in tqdm(range(shuffle_repeat)):
                    pos_ind = np.random.choice(len(svs_list['positive']))
                    data_list_buffer.append(svs_list['positive'][pos_ind])
                    label_list_buffer.append([1,0])        
                    
            elif (self.class_list)[0] == 'negative':
                print('Single Negative')
                cache = 0.00
                for ind in range(len(svs_list['negative'])):
                    cache += self.df.loc[svs_list['negative'][ind]]['neg_area']
                neg_p = []
                for ind in range(len(svs_list['negative'])):
                    neg_p.append(self.df.loc[svs_list['negative'][ind]]['neg_area']/cache)            

                for path_ind in tqdm(range(shuffle_repeat)):
                    neg_ind = np.random.choice(len(svs_list['negative']), p=neg_p)
                    data_list_buffer.append(svs_list['negative'][neg_ind])
                    label_list_buffer.append([0,1])            

        return data_list_buffer, label_list_buffer 
