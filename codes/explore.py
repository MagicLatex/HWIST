import numpy as np
from util import *
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def outlierDetect(X,n_neighbors=20):
    clf = LocalOutlierFactor(n_neighbors=20)
    labels = clf.fit_predict(X)
    return clf,labels
    
    
def main():
    x_names,y_names, widths, heights = load('./sizes.pkl')
    widths_hand = widths[:,0]
    widths_latex = widths[:,1]
    heights_hand = heights[:,0]
    heights_latex = heights[:,1]
    diffs = np.abs(widths_hand-widths_latex)+np.abs(heights_hand-heights_latex)
    max_dist_idx = np.argmax(diffs,)
    max_dist = diffs[max_dist_idx]
    max_dist_hand_name = x_names[max_dist_idx] 
    max_dist_latex_name = y_names[max_dist_idx]
    print(diffs[max_dist_idx])
    print(max_dist_hand_name,max_dist_latex_name)
    wh_hand = np.c_[widths[:,0],heights[:,0]]
    wh_latex = np.c_[widths[:,1],heights[:,1]]
    clf_hand,outlierdetect_hand = outlierDetect(wh_hand,1000)
    clf_latex,outlierdetect_latex = outlierDetect(wh_latex)
    _,outlierdetect_diff = outlierDetect(diffs.reshape((-1,1)))
    num_outlier_hand = np.sum(outlierdetect_hand==-1)
    num_outlier_latex = np.sum(outlierdetect_latex==-1)
    num_outlier_diff = np.sum(outlierdetect_diff==-1)

    max_width_hand, max_height_hand = np.max(widths[:,0]),np.max(heights[:,0])
    min_width_hand, min_height_hand = np.min(widths[:,0]),np.min(heights[:,0])
    max_width_latex, max_height_latex = np.max(widths[:,1]),np.max(heights[:,1])
    min_width_latex, min_height_latex = np.min(widths[:,1]),np.min(heights[:,1])
    max_diff, min_diff = np.max(outlierdetect_diff),np.min(outlierdetect_diff)
    
    xx_hand, yy_hand = np.meshgrid(np.linspace(min_width_hand-10, max_width_hand+10, 1000), np.linspace(min_height_hand-10, max_height_hand+10, 1000))
    xx_latex, yy_latex = np.meshgrid(np.linspace(min_width_latex-10, max_width_latex+10, 1000), np.linspace(min_height_latex-10, max_height_latex+10, 1000))
    xx_diffs = np.linspace(min_diff-10,max_diff+10,1000)
    Z_hand = clf_hand._decision_function(np.c_[xx_hand.ravel(), yy_hand.ravel()])
    Z_latex = clf_latex._decision_function(np.c_[xx_latex.ravel(), yy_latex.ravel()])
    Z_hand = Z_hand.reshape(xx_hand.shape)
    Z_latex = Z_latex.reshape(xx_latex.shape)
    
    #show_distribution  (widths_hand,widths_latex,heights_hand,heights_latex,diffs) 
    print(outlierdetect_hand)
    plt.figure()
    plt.contourf(xx_hand, yy_hand, Z_hand, cmap=plt.cm.Blues_r)
    a = plt.scatter(wh_hand[outlierdetect_hand==1,0][:200], wh_hand[outlierdetect_hand==1,1][:200], c='white',
                    edgecolor='k', s=20)
    b = plt.scatter(wh_hand[outlierdetect_hand==-1,0][:200], wh_hand[outlierdetect_hand==-1,1][:200], c='red',
                    edgecolor='k', s=20)
    plt.title("Handwritten Outlier Detection")
    plt.axis('tight')
    plt.legend([a, b],
               ["normal observations",
                "abnormal observations"],
               loc="upper left")
    plt.grid()
    plt.show()   
    
    plt.figure()
    plt.contourf(xx_latex, yy_latex, Z_latex, cmap=plt.cm.Blues_r)
    a = plt.scatter(wh_latex[outlierdetect_latex==1,0][:200], wh_latex[outlierdetect_latex==1,1][:200], c='white',
                    edgecolor='k', s=20)
    b = plt.scatter(wh_latex[outlierdetect_latex==-1,0][:200], wh_latex[outlierdetect_latex==-1,1][:200], c='red',
                    edgecolor='k', s=20)
    plt.title("Latex Outlier Detection")
    plt.axis('tight')
    plt.legend([a, b],
               ["normal observations",
                "abnormal observations"],
               loc="upper left")
    plt.grid()
    plt.show()  
    
    plt.figure()
    a = plt.scatter(np.ones(200),diffs[outlierdetect_diff==1][:200], c='white',
                    edgecolor='k', s=20)
    b = plt.scatter(np.ones(200),diffs[outlierdetect_diff==-1][:200], c='red',
                    edgecolor='k', s=20)
    plt.title("Diff Outlier Detection")
    plt.axis('tight')
    plt.legend([a, b],
               ["normal observations",
                "abnormal observations"],
               loc="upper left")
    plt.grid()
    plt.show()  

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