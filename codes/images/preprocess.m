hand = imread('hand.png');
latex = imread('latex.png');
hand_gray = rgb2gray(hand);
latex_gray = rgb2gray(latex);
tmp = latex_gray;
tmp(latex_gray>20)= 255;
tmp(latex_gray<200)= 0;
tmp2 = hand_gray;
tmp2(hand_gray>200)= 255;
tmp2(hand_gray<200)= 0;
% tmp = imresize(tmp,size(tmp2));
tmp2 = imresize(tmp2,size(tmp));
final_latex = repmat(tmp,[1,1,3]);
% final_latex = tmp;
final_hand = repmat(tmp2,[1,1,3]);
% final_hand = tmp2;
figure 
imshow(final_latex)
figure
imshow(final_hand)
imwrite(final_latex,'latex_pro.png')
imwrite(final_hand,'hand_pro.png')
% imwrite(final_latex,'latex_pro_1.png')
% imwrite(final_hand,'hand_pro_1.png')