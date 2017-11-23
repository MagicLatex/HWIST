import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread
from PIL import Image
from PIL import Image, ImageChops, ImageOps
import csv
import pickle
import copy


x_dir='./data/handwritten/'
y_dir='./data/latex/'
x_processed_dir='./data/handwritten_processed/'
y_processed_dir='./data/latex_processed/'
train_lst = './im2latex_train.lst'
validate_lst = './im2latex_validate.lst'
test_lst = './im2latex_test.lst'
file_ext = '.png'

def build_dict(lsts,path='./all_dict.pkl'):
    print('Build dictionary ...')
    match = []
    for i in range(len(lsts)):
        cur_lst = lsts[i]
        with open(cur_lst,'rb') as f:
            match = match + [x.decode('utf8').strip('\n') for x in f.readlines()]
    
    match_dict, unmatched = FileNotFoundRemove(match)
    print(('Number of matches %d' % len(match_dict)))
    print(('Number of unfounded files %d' % len(unmatched)))
    save([match_dict,unmatched],path)
    return match_dict,unmatched


def FileNotFoundRemove(match):
    dict = {e.split(' ')[0] + file_ext :e.split(' ')[1] + file_ext for e in match}
    copy_match_dict = copy.deepcopy(dict)
    unmatched = []
    print('Screen dictionary for unfound files ...')
    for k, v in dict.items():
        try:
            #print(k+' '+v)  
            filename_x = x_dir+k
            img_x = Image.open(filename_x)
            
            filename_y = y_dir+v 
            img_y = Image.open(filename_y)
        except FileNotFoundError:
           copy_match_dict.pop(k) 
           unmatched.append(k)
    return copy_match_dict, unmatched
    
def trim(im):
    bg = Image.new(im.mode, im.size, 255)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
        
def rgb2gray(im):
    im_gray = ImageOps.grayscale(im)
    return im_gray

def pad_img(padded_width, padded_height, old_im):
    ifvalid = old_im.size[0]<=padded_width and old_im.size[1]<=padded_height
    if(ifvalid):
        padded_im = Image.new(old_im.mode, (padded_width,padded_height),255)
        padded_im.paste(old_im, (int((padded_width-old_im.size[0])/2), int((padded_height-old_im.size[1])/2)))
    else:
        padded_im = None    
    return padded_im,ifvalid   

def resize_img(resized_width, resized_height, old_im):
    resized_im = old_im.resize((resized_width, resized_height), Image.ANTIALIAS)  
    return resized_im 
    
def get_sizes(match_dict, size_path = './all_size.pkl'):
    print('Get size information from samples ...')
    widths = np.zeros((len(match_dict),2),dtype=np.int)
    heights = np.zeros((len(match_dict),2),dtype=np.int)
    
    x_names = []
    y_names = []
    i = 0

    for k, v in match_dict.items():
        print(k+' '+v)
        filename_x = x_dir+k
        img_x = Image.open(filename_x)
        gray_x = rgb2gray(img_x)
        trimmed_x = trim(gray_x)
        
        filename_y = y_dir+v 
        img_y = Image.open(filename_y)
        gray_y = rgb2gray(img_y)
        trimmed_y = trim(gray_y)

        widths[i,0] = trimmed_x.size[0]
        widths[i,1] = trimmed_y.size[0]
        
        heights[i,0] = trimmed_x.size[1]
        heights[i,1] = trimmed_y.size[1]
        
        x_names.append(k.strip(file_ext))
        y_names.append(v.strip(file_ext))
        
        i = i+1

    valid_mask = np.logical_and(np.all(widths != 0,axis=1),np.all(heights != 0,axis=1))
    widths = widths[valid_mask,:]
    heights = heights[valid_mask,:]
    save([x_names,y_names,widths,heights],size_path) 
    return x_names,y_names,widths, heights

def outlierDetect(x_names,y_names, widths, heights, num_std_away = 1.5, outlier_max_count = 150):
    X = np.c_[widths[:,0],heights[:,0]]
    Y = np.c_[widths[:,1],heights[:,1]]
    print('Outlier Detection ...')
    print('handwritten')
    stds_X = np.std(X,axis=0)
    medians_X = np.median(X,axis=0)
    up_threshs_X = medians_X+num_std_away*stds_X
    down_threshs_X = medians_X-num_std_away*stds_X
    up_mask_X = np.logical_or(X[:,0]>up_threshs_X[0], X[:,1]>up_threshs_X[1])
    down_mask_X = np.logical_or(X[:,0]<down_threshs_X[0], X[:,1]<down_threshs_X[1])
    up_mask_X_idx = np.where(up_mask_X)[0]
    down_mask_X_idx = np.where(down_mask_X)[0]
    overthresh_X = np.sum(X[up_mask_X,:] - up_threshs_X,axis=1)
    belowthresh_X = np.sum(down_threshs_X - X[down_mask_X,:],axis=1)
    notthresh_X = np.hstack((overthresh_X,belowthresh_X))
    print(notthresh_X.shape)
    notthresh_X_idx = np.hstack((up_mask_X_idx,down_mask_X_idx))
    invalid_X_idx = notthresh_X_idx[np.argsort(notthresh_X)[::-1]]
    

    print('latex')
    stds_Y = np.std(Y,axis=0)
    medians_Y = np.median(Y,axis=0)
    up_threshs_Y = medians_Y+num_std_away*stds_Y
    down_threshs_Y = medians_Y-num_std_away*stds_Y
    up_mask_Y = np.logical_or(Y[:,0]>up_threshs_Y[0], Y[:,1]>up_threshs_Y[1])
    down_mask_Y = np.logical_or(Y[:,0]<down_threshs_Y[0], Y[:,1]<down_threshs_Y[1])
    up_mask_Y_idx = np.where(up_mask_Y)[0]
    down_mask_Y_idx = np.where(down_mask_Y)[0]
    overthresh_Y = np.sum(Y[up_mask_Y,:] - up_threshs_Y,axis=1)
    belowthresh_Y = np.sum(down_threshs_Y - Y[down_mask_Y,:],axis=1)
    notthresh_Y = np.hstack((overthresh_Y,belowthresh_Y))
    print(notthresh_Y.shape)
    notthresh_Y_idx = np.hstack((up_mask_Y_idx,down_mask_Y_idx))
    invalid_Y_idx = notthresh_Y_idx[np.argsort(notthresh_Y)[::-1]]

    print('difference')    
    diffs = (X[:,0]-Y[:,0])**2+(X[:,1]-Y[:,1])**2
    stds_diffs = np.std(diffs)
    medians_diffs = np.median(diffs)
    up_threshs_diffs = medians_diffs+num_std_away*stds_diffs
    down_threshs_diffs = medians_diffs-num_std_away*stds_diffs
    up_mask_diffs = diffs>up_threshs_diffs
    down_mask_diffs = diffs<down_threshs_diffs
    up_mask_diffs_idx = np.where(up_mask_diffs)[0]
    down_mask_diffs_idx = np.where(down_mask_diffs)[0]   
    overthresh_diffs = diffs[up_mask_diffs] - up_threshs_diffs
    belowthresh_diffs = down_threshs_diffs - diffs[down_mask_diffs]
    notthresh_diffs = np.hstack((overthresh_diffs,belowthresh_diffs))
    print(notthresh_diffs.shape)
    notthresh_diffs_idx = np.hstack((up_mask_diffs_idx,down_mask_diffs_idx))
    invalid_diffs_idx = notthresh_diffs_idx[np.argsort(notthresh_diffs)[::-1]]
    
    invalid_X_idx,invalid_Y_idx,invalid_diffs_idx = invalid_X_idx[:outlier_max_count],invalid_Y_idx[:outlier_max_count],invalid_diffs_idx[:outlier_max_count]
    invalid_idx = set(np.hstack((invalid_X_idx,invalid_Y_idx,invalid_diffs_idx)))
    invalid_x_names = [x_names[i] for i in invalid_idx]
    invalid_y_names = [y_names[i] for i in invalid_idx]
    save([invalid_x_names,invalid_y_names],'./invalid.pkl')
    return invalid_X_idx,invalid_Y_idx,invalid_diffs_idx
    
def removeOutlierDict(match_dict,invalid_path = './invalid.pkl', dict_path = './dict.pkl'):   
    invalid_x_names,invalid_y_names = load(invalid_path)
    copy_match_dict = copy.deepcopy(match_dict)
    print('Remove outlier for dict...')
    print(('Number of matches before removal: %d' % len(match_dict)))
    for k, v in match_dict.items():
        #print(k+' '+v)
        if(k in invalid_x_names):
            copy_match_dict.pop(k) 
    print(('Number of matches after removal: %d' % len(copy_match_dict)))
    save(copy_match_dict,dict_path) 
    return copy_match_dict

def removeOutlierSize(size_path = './all_size.pkl', invalid_path = './invalid.pkl', resize_path = './size.pkl'):   
    invalid_x_names,invalid_y_names = load(invalid_path)
    x_names,y_names,widths,heights = load(size_path)
    removed_idx = []
    print('Remove outlier for size information...')
    for i in range(len(x_names)):
        #print(x_names[i]+' '+y_names[i])
        if(x_names[i] in invalid_x_names):
            removed_idx.append(i)
    removed_x_names = np.delete(x_names,np.array(removed_idx),0)
    removed_y_names = np.delete(y_names,np.array(removed_idx),0)
    removed_widths = np.delete(widths,np.array(removed_idx),0)
    removed_heights = np.delete(heights,np.array(removed_idx),0)
    save([removed_x_names,removed_y_names,removed_widths,removed_heights],resize_path) 
    return removed_x_names,removed_y_names,removed_widths,removed_heights

def setTargetSize(widths,heights, target_size_path='./target_size.pkl'):
        target_width_x,target_height_x,target_width_y,target_height_y = int(np.median(widths[:,0])),int(np.median(heights[:,0])),int(np.median(widths[:,1])),int(np.median(heights[:,1]))
        save([target_width_x,target_height_x,target_width_y,target_height_y],target_size_path)
        return target_width_x,target_height_x,target_width_y,target_height_y
        
def process(match_dict, widths=None, heights=None, mode='temp', target_size_path='./target_size.pkl'):
    print('Start process data ...')
    x = []
    y = []
    if(mode=='temp'):
        target_width_x,target_height_x,target_width_y,target_height_y = load(target_size_path)
    elif(mode=='save'):
        target_width_x,target_height_x,target_width_y,target_height_y = setTargetSize(widths,heights)  
    print(('target sizes:\n handwritten: %dx%d, latex: %dx%d'% (target_width_x,target_height_x,target_width_y,target_height_y)))
    for k, v in match_dict.items():
        filename_x = x_dir+k
        img_x = Image.open(filename_x)
        gray_x = rgb2gray(img_x)
        trimmed_x = trim(gray_x)
        resized_x = resize_img(target_width_x, target_height_x, trimmed_x)

        filename_y = y_dir+v 
        img_y = Image.open(filename_y)
        gray_y = rgb2gray(img_y)
        trimmed_y = trim(gray_y)
        resized_y = resize_img(target_width_y, target_height_y, trimmed_y)
        if(mode=='temp'):
            x.append(resized_x)
            y.append(resized_y)
        elif(mode=='save'):
            saved_filename_x = x_processed_dir+k
            saved_filename_y = y_processed_dir+v
            resized_x.save(saved_filename_x)  
            resized_y.save(saved_filename_y)
    if(mode=='temp'):
        return x,y
    elif(mode=='save'):
        return

def save(input, dir,protocol = 3):
    pickle.dump(input, open(dir, "wb" ), protocol=protocol)
    return

def load(dir):
    return pickle.load(open(dir, "rb" ))        
        
        
def main():
    match_all_dict,unmatched = build_dict([train_lst,validate_lst,test_lst])
    #match_all_dict,unmatched = load('./all_dict.pkl')
    x_names,y_names,widths, heights = get_sizes(match_all_dict)
    #x_names,y_names,widths,heights = load('./all_size.pkl')    
    outlierDetect(x_names,y_names, widths, heights)
    match_all_dict = removeOutlierDict(match_all_dict)
    #match_all_dict = load('./dict.pkl')
    removed_x_names,removed_y_names,removed_widths,removed_heights = removeOutlierSize()
    #removed_x_names,removed_y_names,removed_widths,removed_heights = load('./size.pkl')
    process(match_all_dict, widths=removed_widths, heights=removed_heights, mode='save')
    
    
if __name__ == '__main__':
    main()