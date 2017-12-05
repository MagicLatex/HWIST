refined = imread('refined.png');
refined_gray = rgb2gray(refined);
tmp(refined_gray>150)= 255;
tmp(refined_gray<150)= 0;
figure
imshow(tmp)