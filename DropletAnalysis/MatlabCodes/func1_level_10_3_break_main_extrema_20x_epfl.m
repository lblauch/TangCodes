%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILENAME: Emulsion breakups main script             
% AUTHOR: Ya, tanglab@stanford   
% DATE: 9/15/2015
% Note: 1. Highlighted drop edges must be shown in raw video (For breakup project)
%          2. Linking part: add 2nd level perimeter matching 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
clear all
close all
% 
% %%%%%%%%%%%% General Setting %%%%%%%%%%%%%%%%%%%%
% file = 'H:\0929\443ulhr_50pl_deg30_w40_h25_oldcam_10x_16000_RhoB_filter2_closeupfps.avi';

% function func1_level_10_3_break_main_extrema_10x(a,b,c)
%     file=a;
    file = 'G:\10262017\600ulhr_1.avi';
    % channel_height = 30;
    seg_method = 0; % Segmentation method, 1(histeq),2(imadjust),3(adapthisteq). Normally use 2.
    numframes = 79245;%17696/2;%67361;%65256%99437;
    area_floor = 300;%400;%50;%10;
    area_ceil = 2500;%2000;
    max_distance = 30;%5; % No larger than this value
    max_delPeri = 20;%20 good for droplet interaction;%3.9;  % For linking: 2nd level perimeter match threshold value
    max_entranceX = 17.5;%17.5;   % Prevent unnecessary counting of the same small drop (disappear and re-appear)
    nproc = 8; % number of processors you want to use
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%% Pre-run %%%%%
    call_distance_distribution = 1;
    call_delPeri_dist = 1;
    call_entranceX_dist = 1;
    call_area_distribution = 1;
    %%%%%%%%%%%%%%%

    % %% Graphic Input Section %%
    obj = VideoReader(file);
    %%%% Upstream Box %%%%
    ui_frame = read(obj,10);
    imshow(ui_frame);
%     [x_box_up,y_box_up] = ginput(2);
    % x_box_up = round([47;130]);%x_box_up);
    % y_box_up = round([14;350]);%y_box_up);

    [row,column]=size(ui_frame);

    x_box_up=[1;column];
    y_box_up=[1;row];

%     x_box_up = round(x_box_up);
%     y_box_up = round(y_box_up);

    %%%% Downstream Box %%%%
%     imshow(ui_frame);
%     [x_box_down,y_box_down] = ginput(2);
    % x_box_down = round([659;748]);%x_box_down);
    % y_box_down = round([115;215]);%y_box_down);

%     x_box_down=[1;column];
%     y_box_down=[1;row];

%     x_box_down = round(x_box_down);
%     y_box_down = round(y_box_down);

    gray = read(obj,1);
    [trow,tcol]=size(gray(:,:,1));
    % load('MASK_500ulhr_70pl_40x_RhB.mat')

    %% Tracking droplets %%
    tic;
    record_up = cell(1,numframes);
    
    %Create mask to remove channel from drop detection
    
 x_mask = [288 288 288 288];
    y_mask = [176 176 176 176];

    
    
    bw2 = poly2mask(x_mask,y_mask,size(gray,1),size(gray,2));
%     bw2 = poly2mask(x_mask,y_mask,size(gray,1),size(gray,2));
    
 x_mask = [288 288 288 288];
    y_mask = [176 176 176 176];

    

    
    bw2 =imcomplement( poly2mask(x_mask,y_mask,size(gray,1),size(gray,2))+bw2);
%     bw2 =imcomplement( poly2mask(x_mask,y_mask,size(gray,1),size(gray,2))+bw2);
    
%     record_down = cell(1,numframes);
    poolobj = parpool('local',nproc); % Initializing parallel computing using local environment
    parfor i = 1:numframes
        % Crop image box for trackDroplets
        gray = im2bw(read(obj,i),0.9).*bw2;

        box_up = gray(y_box_up(1):y_box_up(2),x_box_up(1):x_box_up(2));


        % Track droplets upstream and downstream
        [record_up{i},e_up{i},h_up{i},pl_up{i},ori_up{i},bw_view{i}] = level_10_4_trackDrops_extrema_rhodB(box_up,area_floor,area_ceil,seg_method,bw2);   % Upstream use a larger area_floor value
%         [record_down{i},e_down{i},h_down{i},pl_down{i}] = level_10_4_trackDrops_extrema(box_down,area_floor,area_ceil,seg_method);

    end

    delete(poolobj);
    toc



    %% Linking droplets %%
    tic;
    [record_up,minDistance_up,minDelPeri_up,minEntry_up] = level_10_5_linkDrops_extrema(record_up,max_distance,max_delPeri,max_entranceX,numframes);
%     [record_down, minDistance_down,minDelPeri_down,minEntry_down] = level_10_5_linkDrops_extrema(record_down,max_distance,max_delPeri,max_entranceX,numframes);
    toc

    %% Pre-run diagnostics %%
    if call_distance_distribution == 1
        minDistance_up = [cell2mat(cellfun(@(x) x(:), minDistance_up, 'uni', 0)')]';
        figure
        hist(minDistance_up,0:0.2:40)
        title('up box droplet linking distance distribution')
        
%             minDistance_down = [cell2mat(cellfun(@(x) x(:), minDistance_down, 'uni', 0)')]';
%             figure
%             hist(minDistance_down,0:0.2:40)
%             title('down box droplet linking distance distribution')
    end

    if call_delPeri_dist == 1;
        minDelPeri_up = [cell2mat(cellfun(@(x) x(:), minDelPeri_up, 'uni', 0)')]';
        figure
        hist(minDelPeri_up,0:0.1:40)
        title('up box droplet perimeter match distribution')
        
%             minDelPeri_down = [cell2mat(cellfun(@(x) x(:), minDelPeri_down, 'uni', 0)')]';
%             figure
%             hist(minDelPeri_down,0:0.1:40)
%             title('down box droplet perimeter match distribution')
    end

    if call_entranceX_dist == 1
        minEntry_up = [cell2mat(cellfun(@(x) x(:), minEntry_up, 'uni', 0)')]';
        minEntry_up(minEntry_up==0) = [];
        figure
        hist(minEntry_up,0:0.2:100)
        title('up box just-entering droplet x location')
        
%             minEntry_down = [cell2mat(cellfun(@(x) x(:), minEntry_down, 'uni', 0)')]';
%             minEntry_down(minEntry_down==0) = [];
%             figure
%             hist(minEntry_down,0:0.2:100)
%             title('down box just-entering droplet x location')
    end

    if call_area_distribution == 1
        area_up = level_10_7_dropArea( record_up );
        figure
        hist(area_up,0:1:5000)
        title('up box droplet area distribution')
        
%             area_down = level_10_7_dropArea( record_down );
%             figure
%             hist(area_down,0:1:1000)
%             title('down box droplet area distribution')
    end
    
%     save(b)
    
    
% end

load gong.mat;
soundsc(y);
1
% LeadingEdge
boundarydetection_dropletoffset

2
% save('cs231n_600ulhr_1.mat')
% save('Conc_Emul_0p6mh_20x_10012017.mat')
% save('Conc_Emul_862ulhr_50pL_DIWdrop_H25um_W30um_15deg_20x.mat')