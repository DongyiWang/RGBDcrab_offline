clc
clear all
close all
% root = '../data/Recon/';
% dir_root = dir(strcat(root, '*.bmp'));
% for i = 1:length(dir_root)
%      Im_single = imread(strcat(root, dir_root(length(dir_root) + 1 - i).name));
%      Im_cube(:,:,i) = Im_single(100:580, 100:920);
% end
% Im_cube = imresize3(Im_cube, 1);
% Im_cube = medfilt3(Im_cube);
% Im_BW = imbinarize(Im_cube, 10/255);
% stats = regionprops3(Im_BW,'Volume');
% stats_Volume = stats.Volume;
% Im_BW = bwareaopen(Im_BW, max(stats_Volume));
% Im_BW = imfill(Im_BW,'holes');
% save('../data/Crab_model.mat', 'Im_BW');

Crab_3D = load('../data/Crab_model.mat');
Crab_3D = Crab_3D.Im_BW;
Crab_3D = imresize3(uint8(Crab_3D)*255, 0.3368);
Crab_3D = (Crab_3D > 0);
% myvolshow = volshow(Crab_3D);
Crab_2D_binary = squeeze(sum(Crab_3D, 1));
Crab_2D_binary = imrotate(Crab_2D_binary, 270);
Crab_2D_binary = (Crab_2D_binary > 0);
%Crab_2D_from_3D = imresize(Crab_2D_from_3D, [ceil(size(Crab_2D_from_3D, 1)*0.3314), ceil(size(Crab_2D_from_3D, 2)*0.3314)]);
Crab_2D = imread('../data/3.tif');
Crab_2D_label = imread('../data/3_label.tif');
Crab_2D_label = (Crab_2D_label(:,:,1)==255) & (Crab_2D_label(:,:,2)==0) & (Crab_2D_label(:,:,3)==0);
x_start = 514;
y_start = 264;

for i = 1:size(Crab_3D, 2)
    for j = 1:size(Crab_3D, 3)
        Height_single = Crab_3D(:, i, j);
        Height_idx = 0;
        if sum(Height_single) > 0
            Height_idx = find(Height_single);
            Height_idx = size(Crab_3D, 1) - min(Height_idx);
        end           
        Height_map(i, j) = Height_idx;
    end
end
Height_map = imrotate(Height_map, 270);
figure 
imshow(uint8(Height_map))
imwrite(uint8(Height_map), '../data/Height_map.tif')
for i = y_start:y_start+size(Crab_2D_binary, 1)-1
    for j = x_start:x_start+size(Crab_2D_binary, 2)-1
        if Crab_2D_label(i, j) == 1 && Crab_2D_binary(i-y_start+1, j-x_start+1) == 1
            Crab_2D(i, j, 1) = 0;
            Crab_2D(i, j, 3) = 0;
            Crab_2D(i, j, 2) = Height_map(i-y_start+1, j-x_start+1);
        end
    end
end

figure 
imshow(Crab_2D)
imwrite(Crab_2D, '../data/Crab_2D.tif')
%figure
%volshow(Crab_3D)