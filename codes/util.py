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
        #print(k+' '+v)
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
    
def removeOutlierDict(match_dict,invalid_path = './invalid.pkl', dict_path = './dict.pkl'):   
    invalid_x_names,invalid_y_names = load(invalid_path)
    copy_match_dict = copy.deepcopy(match_dict)
    print('Remove outlier for dict...')
    print(('Number of matches before removal: %d', len(match_dict)))
    for k, v in match_dict.items():
        #print(k+' '+v)
        if(k in invalid_x_names):
            copy_match_dict.pop(k) 
    print(('Number of matches after removal: %d', len(copy_match_dict)))
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
    print(('target sizes:\n handwritten: %dx%d, latex: %dx%d'% target_width_x,target_height_x,target_width_y,target_height_y))
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
    #match_all_dict,unmatched = build_dict([train_lst,validate_lst,test_lst])
    match_all_dict,unmatched = load('./all_dict.pkl')
    #x_names,y_names,widths, heights = get_sizes(match_all_dict)
    x_names,y_names,widths,heights = load('./all_size.pkl')
    #match_all_dict = removeOutlierDict(match_all_dict)
    match_all_dict = load('./dict.pkl')
    #removed_x_names,removed_y_names,removed_widths,removed_heights = removeOutlierSize()
    removed_x_names,removed_y_names,removed_widths,removed_heights = load('./size.pkl')
    process(match_all_dict, widths=removed_widths, heights=removed_heights, mode='save')
    
    
if __name__ == '__main__':
    main()