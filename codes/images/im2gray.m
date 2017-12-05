function out=im2gray(im)
if size(im,3)>1
	out=im2double(rgb2gray(im));
else
	out=im2double(im);
end
end