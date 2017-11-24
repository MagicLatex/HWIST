import numpy as np
from util import *
import matplotlib.pyplot as plt
import pickle

def outlierDetect(X, Y, num_std_away = 1.5, outlier_max_count = 150):
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
 
 
 
 
x_names,y_names, widths, heights = load('./all_size.pkl')  
def main():
    widths_hand = widths[:,0]
    widths_latex = widths[:,1]
    heights_hand = heights[:,0]
    heights_latex = heights[:,1]
    diffs = (widths_hand-widths_latex)**2+(heights_hand-heights_latex)**2
    max_dist_idx = np.argmax(diffs,)
    max_dist = diffs[max_dist_idx]
    max_dist_hand_name = x_names[max_dist_idx] 
    max_dist_latex_name = y_names[max_dist_idx]
    print(diffs[max_dist_idx])
    print(max_dist_hand_name,max_dist_latex_name)
    wh_hand = np.c_[widths[:,0],heights[:,0]]
    wh_latex = np.c_[widths[:,1],heights[:,1]]
    invalid_X_idx,invalid_Y_idx,invalid_diffs_idx = outlierDetect(wh_hand,wh_latex)
    invalid_idx = set(np.hstack((invalid_X_idx,invalid_Y_idx,invalid_diffs_idx)))
    print(invalid_idx)
    
    invalid_x_names_X =[x_names[i] for i in invalid_X_idx]
    invalid_y_names_X =[y_names[i] for i in invalid_X_idx]
    invalid_x_names_Y =[x_names[i] for i in invalid_Y_idx]
    invalid_y_names_Y =[y_names[i] for i in invalid_Y_idx]
    invalid_x_names_diffs =[x_names[i] for i in invalid_diffs_idx]
    invalid_y_names_diffs =[y_names[i] for i in invalid_diffs_idx]
    invalid_x_names = [x_names[i] for i in invalid_idx]
    invalid_y_names = [y_names[i] for i in invalid_idx]
    print('invalid for hand')
    print(len(invalid_x_names_X))
    print(invalid_x_names_X[-100:])
    print(invalid_y_names_X[-100:])
    print('invalid for latex')
    print(len(invalid_x_names_Y))
    print(invalid_x_names_Y[-100:])
    print(invalid_y_names_Y[-100:])
    print('invalid for diffs')
    print(len(invalid_x_names_diffs))
    print(invalid_x_names_diffs[-100:])
    print(invalid_y_names_diffs[-100:])
    print('invalid for all')
    print(invalid_x_names)
    print(invalid_y_names)    
    #show_distribution(widths_hand,widths_latex,heights_hand,heights_latex,diffs) 


def show_distribution(widths_hand,widths_latex,heights_hand,heights_latex,diffs):
    plt.figure()
    plt.hist(widths_hand, 'auto', normed=1, facecolor='red', alpha=0.5, label='handwritten')
    plt.hist(widths_latex, 'auto', normed=1, facecolor='blue', alpha=0.5, label='latex')
    plt.title('widths histogram')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.hist(heights_hand, 'auto', normed=1, facecolor='red', alpha=0.5, label='handwritten')
    plt.hist(heights_latex, 'auto', normed=1, facecolor='blue', alpha=0.5, label='latex')
    plt.title('heights histogram')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.hist(diffs, 'auto', normed=1, facecolor='red', alpha=0.75)
    plt.title('area differences histogram')
    plt.grid()
    plt.show()  
    return
if __name__ == '__main__':
    main()