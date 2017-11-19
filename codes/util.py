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

def build_dict(lsts,path='./dict.pkl'):
    print('Build dictionary ...')
    match = []
    for i in range(len(lsts)):
        cur_lst = lsts[i]
        with open(cur_lst,'rb') as f:
            match = match + [x.decode('utf8').strip('\n') for x in f.readlines()]
    
    match_dict, unmatched = FileNotFoundRemove(match)
    print(len(match_dict))
    save([match_dict,unmatched],'./all_dict.pkl')
    return match_dict,unmatched


def FileNotFoundRemove(match):
    dict = {e.split(' ')[0] + file_ext :e.split(' ')[1] + file_ext for e in match}
    copy_match_dict = copy.deepcopy(dict)
    unmatched = []
    print(len(dict))
    for k, v in dict.items():
        try:
            print(k+' '+v)  
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
    
def get_sizes(match_dict,path = './meta.csv'):
    print('Get sizes samples ...')
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
    save([x_names,y_names,widths,heights],'./sizes.pkl') 
    return x_names,y_names,widths, heights  
    
def process(match_dict, padded_width_x, padded_height_x, padded_width_y, padded_height_y, ifsave=True):
    print('Start process data ...')
    outofbound = []
    for k, v in match_dict.items():
        filename_x = x_dir+k
        img_x = Image.open(filename_x)
        gray_x = rgb2gray(img_x)
        trimmed_x = trim(gray_x)
        padded_x,ifvalid_x = pad_img(padded_width_x, padded_height_x, trimmed_x)

        filename_y = y_dir+v 
        img_y = Image.open(filename_y)
        gray_y = rgb2gray(img_y)
        trimmed_y = trim(gray_y)
        padded_y,ifvalid_y = pad_img(padded_width_y, padded_height_y, trimmed_y)

        if(ifvalid_x&ifvalid_y):
            print(k+' '+v)
            saved_filename_x = x_processed_dir+k
            saved_filename_y = y_processed_dir+v
            if(ifsave):
                padded_x.save(saved_filename_x)  
                padded_y.save(saved_filename_y)  
        else:
            outofbound.append(k)
    save(outofbound,'./outofbound.pkl')    
    return

def save(input, dir,protocol = 3):
    pickle.dump(input, open(dir, "wb" ), protocol=protocol)
    return

def load(dir):
    return pickle.load(open(dir, "rb" ))        
        
        
def main():
    match_all_dict,unmatched=build_dict([train_lst,validate_lst,test_lst],'./all_dict.pkl')
    #match_all_dict,unmatched = load('./all_dict.pkl')
    x_names,y_names,widths, heights = get_sizes(match_all_dict,path = './meta.csv')
    #x_names,y_names,widths,heights = load('./sizes.pkl')
    padded_width, padded_height = np.max(widths),np.max(heights)
    #padded_width_x, padded_height_x, padded_width_y, padded_height_y = np.max(widths[:,0]), np.max(heights[:,0]), np.max(widths[:,1]), np.max(heights[:,1]) 
    print(padded_width,padded_height)
    #process(match_all_dict, padded_width, padded_height, padded_width, padded_height)
    
    
if __name__ == '__main__':
    main()