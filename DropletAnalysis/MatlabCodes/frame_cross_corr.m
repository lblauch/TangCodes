%%%%%%%
close all;
clear all;

vidObj = VideoReader('LEO_9_Frame_5615_50pL_600ulhr_noBreak.avi');

N=35;  %number of frames to read
corrMax=zeros(1,N); %% setup cross correlation vector to fill in

% while hasFrame(vidObj)
for i=1:N
    if i==1
        f1 = readFrame(vidObj);
        f1=f1(33:181,88:335,1);
        imagesc(f1); colormap gray;
    end
        f = readFrame(vidObj);
        f=f(33:181,88:335,1);

    %% calculate xcorr
    crr = xcorr2(f1, f);
    [ssr,snd] = max(crr(:));
    corrMax(i)=ssr;
end

figure; plot(corrMax);


