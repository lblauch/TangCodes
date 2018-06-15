%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILENAME: function version of level_1_1              
% AUTHOR: Ya, tanglab@stanford   
% DATE: 9/10/2015
% Note: Tuned for breakup project. imfill is on. Optimized imfill placement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [record,e,h,pl,ori,bw] = level_10_4_trackDrops_extrema_rhodB( box,area_floor,area_ceil,seg_method,mask )
%% Noise filter on raw image %%
%     box = imguidedfilter(box);
%     box = medfilt2(box);
%     box = wiener2(box);
    
%% Increase contrast level %%
if seg_method == 1
    contrasted = histeq(box);
elseif seg_method == 2
    contrasted = imadjust(box);
elseif seg_method ==3
    contrasted = adapthisteq(box);
elseif seg_method == 0
    contrasted = box;
end

%% Grayscale to B&W %%
% contrasted=imcomplement(contrasted);
thres = graythresh(contrasted);  % Calculate threshold value for binary image conversion
% thres = 0.5;
% bw = contrasted;
bw = im2bw(contrasted,0.9);   % Convert to B&W image
% bw = imcomplement(bw);
% bw=bw.*mask;
% bw = imcomplement(bw);
% SE=strel('square',6);
% bw=imerode(bw,SE);


%% Fill droplet interior %%
bw = imfill(bw,'holes');    % Note to add 'holes', otherwise the code will stop here

%% Remove regions with area larger than area_ceil or smaller than area_floor %%
bw = xor( bwareaopen(bw,area_floor,4),  bwareaopen(bw,area_ceil,4) );     % Remove regions with area larger than area_ceil or smaller than area_floor. The default connectivity of bwareaopen is 8. 

%% Determine whole droplets %%
% horizontal not whole droplets
[label,~] = bwlabel(bw,4);
left = label(:,1);
left(left==0) = [];
right = label(:,end);
right(right==0) = [];
del_h = unique([left;right]);
for j = 1:length(del_h)
    label( label==del_h(j) ) = 0; 
end
% % vertical not whole droplets
% up = label(1,:);
% up(up==0) = [];
% down = label(end,:);
% down(down==0) = [];
% del_v = unique([up,down]);
% for j = 1:length(del_v)
%     label( label==del_v(j) ) = 0;
% end

%% Reinforce 4-connectivity and define connected objects %%
CC = bwconncomp(label,4); 

%% Calculate geometry information of each effective drops
c = regionprops(CC,'Centroid');
centroid = cat(1,c.Centroid);
a = regionprops(CC,'Area');    
area = cat(1,a.Area);
p = regionprops(CC,'Perimeter');
perimeter = cat(1,p.Perimeter);
MAJL = regionprops(CC,'MajorAxisLength');
major = cat(1,MAJL.MajorAxisLength);
MINL = regionprops(CC,'MinorAxisLength');
minor = cat(1,MINL.MinorAxisLength);
n = CC.NumObjects;
id = transpose(1:n);
e=regionprops(CC,'Extrema');
h=regionprops(CC,'ConvexHull');
pl=regionprops(CC,'PixelList');
ori=regionprops(CC,'Orientation');
%extrema=cat(1,e.Extrema);
record= cat(2,id,centroid,area,perimeter,major,minor);



end

