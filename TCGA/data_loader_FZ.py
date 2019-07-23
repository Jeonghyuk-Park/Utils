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
BLUEPRINT

 - Preprocess 클래스 
학습을 위한 데이터 전처리, online sampling 할 좌표를 pickle로 저장

 - Load 클래스
tf dataset API를 위한 데이터 로딩


 - 2class
18/18 x 10 class 분량, 360 슬라이드를 버퍼로 옮기고
class 상관없이 positive, negative 만 

버퍼는 20을 사용할 떄, 각 클래스별 p/n 이 한번씩 다 들어갈 수 있도록 제작
총 360 path 들어가게
"""

__base_path__ = '/ssd5/NIH/access_coordinate_pickle/'
#글로벌 변수로 사용됨

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

class Preprocess:
    """
    Class Preprocess
    tissue가 존재하는 coordinate를 미리 계산하여 export하는 함수들이 포함됨 (access_coordinate)
    
    svs_paths = glob.glob('/ssd4/NIH/LUAD_FFPE/*/*.svs')
    print(len(svs_paths))
    Preprocess = data_loader.Preprocess(svs_paths)
    Preprocess.access_coordinate()
    
    """
    
    def __init__(self, img_path_list, level = 1, threshold = .8):
    
        self.img_path_list = img_path_list
        self.level = level
        self.threshold = threshold
    
    def access_coordinate(self):
        """
        Input : abs_path of svs image (string), 
                level (default = 1),
                threshold (default = 0.8)
        Return : None
        해당하는 이미지의 바이너리 마스크 이미지를 계산 후 np.where 파일을, 파일명.pickle로 저장
        저장할 path는 함수 내부에 내장되어있으나 외부에서 입력받게 해도 됨
        """
        
        level = self.level
        threshold = self.threshold
        img_path_list = self.img_path_list
        
        def calc_fn(img_path, level):
            level = self.level
            img = open_rgb(img_path = img_path, level = level)            
            
            if img is False: # image open이 안되는 경우 일단 에러 로그로 남김
                filename = '.'.join(img_path.split('/')[-1].split('.')[:-1])
                return 0, filename
            
            else:
                img = img.astype(np.float32)/255 
                x, y = int(img.shape[0]/16), int(img.shape[1]/16) #계산 편의상 1/16 사이즈로 리사이즈함
                img = resize(img, (x, y, 3))
                img_mask = img.mean(axis=-1) < threshold
                access_coordinate_list = np.where(img_mask)

                global __base_path__
                filename = '.'.join(img_path.split('/')[-1].split('.')[:-1])
                save_path = __base_path__ + filename + '.pickle'
                #파일 이름 그대로 붙여서 base_path 하위에 pickle 저장
                with open(save_path, 'wb') as handle:
                    pickle.dump(access_coordinate_list, handle)
                #주의 : 디렉토리가 없으면 저장이 안되므로 수동으로 디렉토리 생성 요망. 
                #버그발생 방지를 위해서 os.makedir 을 사용하지 않음
                return [len(access_coordinate_list[0]), img_mask.sum()], filename #access_coordinate_list 리스트의 길이와 저장된 path 출력
        
        t0 = time.time()
        log = {}
        
        for ind, img_path in enumerate(img_path_list):
            tissue_size, filename = calc_fn(img_path = img_path, level = self.level)
            print(ind+1, '/', len(img_path_list), ' time passed (s) : ', time.time()-t0)
            print(tissue_size, filename)
            log[filename] = tissue_size
                    
        return log
            
        
        
        
        
            
class Load:
    """
    svs list를 받으면, pickle 을 미리 메모리에 올리고
    patch를 생성하는 하위 함수를 포함함
    """
    def __init__(self, dataset_ind = 0, includeTest = False):
        """
        """
        self.dataset_ind = dataset_ind
        self.includeTest = includeTest
        
        def frozen_tissue(cancer_class):
            """
            cancer_class = 'STAD', 'LUAD', 'LUSC', 'COAD', 'PRAD'
            """
            img_path_list = glob.glob('/nfs2/jhpark/NIH/{}/*/*.svs'.format(cancer_class))
            assert len(img_path_list) > 0
            #적어도 1개 이상의 원소를 포함하는 리스트인지 확인. 에러시 암종 이름을 정확하게 입력했는지 확인요망
        
            def frozen_slide_list(img_path_list):
                """
                3자리 태그의 가운데가 S인 슬라이드가 냉동 검체
                path를 모아서 list로 반환
                """
                list_buffer = []

                for elem in img_path_list:
                    tag = elem.split('-')[-5].split('.')[0]
                    if tag[1] == 'S': #태그 중간이 S인것만 추가함
                        list_buffer.append(elem)

                return list_buffer

            def FFPE_slide_list(img_path_list):
                """
                3자리 태그의 가운데가 S가 아닌 FFPE슬라이드
                path를 모아서 list로 반환
                """
                list_buffer = []

                for elem in img_path_list:
                    tag = elem.split('-')[-5].split('.')[0]
                    if tag[1] != 'S': #태그 중간이 S가 아닌 것만 추가함
                        list_buffer.append(elem)

                return list_buffer
    
            def pos_neg_divide(list_buffer):
                """
                냉동 검체 데이터 중 양성/음성 데이터로 구분하는 fn
                """

                positive_list = []
                negative_list = []

                for elem in list_buffer:
                    tumor_tag = elem.split('-')[-7]
                    if tumor_tag[:2] == '01':
                        positive_list.append(elem)
                    elif tumor_tag[:2] == '11':
                        negative_list.append(elem)

                return positive_list, negative_list
            
            def ispickle(img_path_list):
                list_buffer = []
                global __base_path__
                for img_path in img_path_list:
                    filename = img_path.split('/')[-1][:-4]
                    if os.path.isfile(__base_path__ + filename + '.pickle') is True:
                        list_buffer.append(img_path)
                return list_buffer
        
            frozen_slide_positive = ispickle(pos_neg_divide(frozen_slide_list(img_path_list))[0])
            frozen_slide_negative = ispickle(pos_neg_divide(frozen_slide_list(img_path_list))[1])
            return frozen_slide_positive, frozen_slide_negative

        all_classes = ['STAD', 'COAD', 'LUAD', 'LUSC', 'UCEC', 'BLCA', 'PRAD', 'THCA', 'BRCA', 'LIHC','OV', 'KIRC']
        
        img_path_list_buffer_pos = []
        img_path_dict_buffer_pos = {}
        img_path_list_buffer_neg = []
        img_path_dict_buffer_neg = {}
        
        for class_name in all_classes:
            img_path_list_buffer_pos += frozen_tissue(class_name)[0][:18]
            img_path_dict_buffer_pos[class_name] = frozen_tissue(class_name)[0][:18]
            img_path_list_buffer_neg += frozen_tissue(class_name)[1][:18]
            img_path_dict_buffer_neg[class_name] = frozen_tissue(class_name)[1][:18]
        
        self.img_path_list_pos = img_path_list_buffer_pos
        self.img_path_dict_pos = img_path_dict_buffer_pos
        self.img_path_list_neg = img_path_list_buffer_neg
        self.img_path_dict_neg = img_path_dict_buffer_neg        

        def preloading(img_path_list):
            
            global __base_path__
            
            coord_mem_buffer = {}            
            pickles = []
            
            assert img_path_list[0].split('/')[-1][-3:] == 'svs'
            #list 의 원소의 확장자가 svs여야함.
            
            for img_path in img_path_list:
                pickle_path = __base_path__ + img_path.split('/')[-1][:-4] + '.pickle'
                if os.path.isfile(pickle_path) is True:
                    pickles.append(pickle_path)
                else:
                    print(pickle_path + ' is not exist!')
            #pickles = glob.glob(__base_path__ + '*.pickle')
            assert len(img_path_list) == len(pickles)
            
            for pickle_ in tqdm(pickles):
                with open(pickle_, 'rb') as f:
                    data = pickle.load(f)
                filename = pickle_.split('/')[-1][:-7]
                coord_mem_buffer[filename] = data

            return coord_mem_buffer
        nfs_path = '/nfs/ywkim/GreenCross/GitHub/NIHdataset/dataset_pickle/'        
        def path_convert(path):
            return '/ssd5/NIH/' + '/'.join(path.split('/')[-3:])
        
        preload_buffer = []
        
        for path_ind in range(18):
            for class_name in all_classes:
                with open(nfs_path + 'dataset_entity_{}_{}.pickle'.format(class_name, self.dataset_ind), 'rb') as handle:
                    list_dict = pickle.load(handle)
                preload_buffer.append(path_convert(list_dict['train_pos'][path_ind]))
                preload_buffer.append(path_convert(list_dict['train_neg'][path_ind]))
                if includeTest == True:
                    preload_buffer.append(path_convert(list_dict['test_pos'][path_ind]))
                    preload_buffer.append(path_convert(list_dict['test_neg'][path_ind]))
        
        self.preload_buffer = preload_buffer
        self.coord_mem = preloading(preload_buffer)
        print('Memory Loading Done')
    
    def patch_gen_fn(self, file_path, label, isAscii = True):
        """
        파일path을 받았을 때, 파일명을 키로 좌표 리스트를 받아서, 랜덤한 좌표를 선택하고, 패치를 생성하는 함수
        coord_mem_load의 리턴 저장값을 coord_mem에 넣고 동작
        coord_mem_buffer에서 랜덤하게 선택
        """
        #pos_or_neg = np.random.randint(2)
        #if pos_or_neg == 0:
        #    file_path = self.pos_path_list[np.random.randint(len(self.pos_path_list))]
        #    label = [1,0]
        #elif pos_or_neg == 1:
        #    file_path = self.neg_path_list[np.random.randint(len(self.neg_path_list))]
        #    label = [0,1]
        #else:
        #    print("error")
        
        level = 1
        patch_size = 299
        if isAscii == True:
            file_path = file_path.decode('ascii') #file path 가 binary로 들어오므로 decoding 필요, dataset mapping 전용
    
        coord_mem = self.coord_mem 
        file_name = file_path.split('/')[-1][:-4] # path에서 디렉토리와 확장자 제거한 filename 만 반환
        file_coord_list = coord_mem[file_name] # coord_mem dict에서, 좌표 list를 반환
        randint = np.random.randint(len(file_coord_list[0])) # 좌표 리스트 랜덤 indice 선택
        i,j = file_coord_list[0][randint], file_coord_list[1][randint]

    
        def open_patch(img_path, 
                       coords, 
                       level=level, 
                       patch_size=patch_size, 
                       coord_resized_factor = 16): 
            """
            좌표 구할 때 리사이즈 팩터 기본값은 16
            i, j 를 역으로 ([1],[0]) 으로 해야 동작함
            현재 이 코드는 레벨 1에서 동작하도록, 리전 영역(레벨 0 기준)에 4배를 곱하게 되어있음        
            """
            coord_i = int(coord_resized_factor*coords[1])
            coord_j = int(coord_resized_factor*coords[0])

            try:
                slide_img = openslide.OpenSlide(img_path)
                read_region_coords = [coord_i*4+np.random.randint(coord_resized_factor*4),
                                      coord_j*4+np.random.randint(coord_resized_factor*4)]
                #좌표를 레벨0기준에서 받기 때문에, 4배 곱하고, 랜덤값으로 리사이즈에 의한 효과를 보정
                rgb_img = np.array(slide_img.read_region(read_region_coords, level, [patch_size,patch_size]))[:,:,:3]
                return rgb_img
            except:
                print('cannot open', img_path)
                return False

        patch = open_patch(file_path, level=1, coords=[i,j], patch_size=patch_size)
        label = np.array(label, dtype=np.float32)

        return np.array(patch/255, dtype=np.float32), label


    def list_gen_fn(self, shuffle_repeat = 0):
        """
        미리 저장된 pickle로부터 데이터셋을 불러옴.
        """
        import random

        def path_convert(path):
            return '/ssd5/NIH/' + '/'.join(path.split('/')[-3:])

        nfs_path = '/nfs/ywkim/GreenCross/GitHub/NIHdataset/dataset_pickle/'

        data_list_buffer = []
        label_list_buffer = []

        all_classes = ['STAD', 'COAD', 'LUAD', 'LUSC', 'UCEC', 'BLCA', 'PRAD', 'THCA', 'BRCA', 'LIHC','OV', 'KIRC']

        dataset_dict = {}
        for class_name in all_classes:
            with open(nfs_path + 'dataset_entity_{}_{}.pickle'.format(class_name, self.dataset_ind), 'rb') as handle:
                list_dict = pickle.load(handle)
            dataset_dict[class_name, 'positive'] = list_dict['train_pos']
            dataset_dict[class_name, 'negative'] = list_dict['train_neg']      

        for path_ind in range(18):
            for class_name in all_classes:
                data_list_buffer.append(path_convert(dataset_dict[class_name, 'positive'][path_ind]))
                label_list_buffer.append([1,0])
                data_list_buffer.append(path_convert(dataset_dict[class_name, 'negative'][path_ind]))
                label_list_buffer.append([0,1])

        for ind in range(shuffle_repeat):

            for path_ind in range(18):
                for class_name in all_classes:
                    pos_list = dataset_dict[class_name, 'positive']
                    random.shuffle(pos_list)
                    data_list_buffer.append(path_convert(pos_list[path_ind]))
                    label_list_buffer.append([1,0])
                    neg_list = dataset_dict[class_name, 'negative']
                    random.shuffle(neg_list)
                    data_list_buffer.append(path_convert(neg_list[path_ind]))
                    label_list_buffer.append([0,1])

        return data_list_buffer, label_list_buffer

    def list_gen_fn_v2(self, class_list, shuffle_repeat = 0):
        """
        미리 저장된 pickle로부터 데이터셋을 불러옴.
        class_list 는 ['STAD'] 와 같이 리스트 안의 str 형태로 입력하세요
        """
        import random

        def path_convert(path):
            return '/ssd5/NIH/' + '/'.join(path.split('/')[-3:])

        nfs_path = '/nfs/ywkim/GreenCross/GitHub/NIHdataset/dataset_pickle/'

        data_list_buffer = []
        label_list_buffer = []

        dataset_dict = {}
        for class_name in class_list:
            with open(nfs_path + 'dataset_entity_{}_{}.pickle'.format(class_name, self.dataset_ind), 'rb') as handle:
                list_dict = pickle.load(handle)
            dataset_dict[class_name, 'positive'] = list_dict['train_pos']
            dataset_dict[class_name, 'negative'] = list_dict['train_neg']      

        for path_ind in range(18):
            for class_name in class_list:
                data_list_buffer.append(path_convert(dataset_dict[class_name, 'positive'][path_ind]))
                label_list_buffer.append([1,0])
                data_list_buffer.append(path_convert(dataset_dict[class_name, 'negative'][path_ind]))
                label_list_buffer.append([0,1])

        for ind in range(shuffle_repeat):

            for path_ind in range(18):
                for class_name in class_list:
                    pos_list = dataset_dict[class_name, 'positive']
                    random.shuffle(pos_list)
                    data_list_buffer.append(path_convert(pos_list[path_ind]))
                    label_list_buffer.append([1,0])
                    neg_list = dataset_dict[class_name, 'negative']
                    random.shuffle(neg_list)
                    data_list_buffer.append(path_convert(neg_list[path_ind]))
                    label_list_buffer.append([0,1])

        return data_list_buffer, label_list_buffer

    def debug(self):
        return self.coord_mem
