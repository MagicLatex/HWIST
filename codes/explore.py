import numpy as np
from util import *
import matplotlib.pyplot as plt
import pickle

 

def main():
    x_names,y_names, widths, heights = load('./all_size.pkl')  
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

    invalid_X_idx,invalid_Y_idx,invalid_diffs_idx = outlierDetect(x_names,y_names, widths, heights)
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