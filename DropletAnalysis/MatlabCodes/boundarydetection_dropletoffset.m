% DropletEdge=[];

% load('EdgeBoundaryDetectionTest.mat')
nproc = 8;
poolobj = parpool('local',nproc);

% Matrix pl_up is a cell that stores the pixel list of drops
% This code extracts the drop boundary and also its leading edge by looking
% for the largest value in x-direction for each drop



numframes=79245;
parfor i=1:numframes
    
    
    [r,c]=size(pl_up{i});
    for j=1:r
        
        [r2,c2]=size(pl_up{i}(j).PixelList);
        I=[];
        I=zeros(max(pl_up{i}(j).PixelList(:,1)),max(pl_up{i}(j).PixelList(:,2)));
        
        for k=1:r2
            I(pl_up{i}(j).PixelList(k,1),pl_up{i}(j).PixelList(k,2))=1;
        end
        
        
        
        DropletEdge{i}(j,1)=bwboundaries(I);


    end

end


parfor i=1:numframes
        [r,c]=size(DropletEdge{i});
        for j=1:r
            LeadingEdge{i}(j,1)=max(DropletEdge{i}{j}(:,1));
%             DiscreteR{i}{j,1}(:,1)=sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2);
%             Circularity{i}(j,1) = 2.768*(0.361-sqrt(2)*std (sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2)/(max(sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2)))));
        end
        
end

parfor i=1:numframes
        [r,c]=size(DropletEdge{i});
        for j=1:c
%             LeadingEdge{i}(j,1)=max(DropletEdge{i}{j}(:,1));
            DiscreteR{i}{j,1}(:,1)=sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2);
            Spreadness{i}{j,1}=std (sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2)/(max(sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2))));
            Circularity{i}(j,1) = 2.768*(0.361-sqrt(2)*std (sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2)/(max(sqrt((DropletEdge{i}{j}(:,1)-record_up{i}(j,2)).^2+(DropletEdge{i}{j}(:,2)-record_up{i}(j,3)).^2)))));
        end
        
end
delete(poolobj); 
% [r,c]=size(B{1,1});
% newB=zeros(max(B{1,1}(:,1))+100,max(B{1,1}(:,2))+100);
% 
% for i=1:r
%     newB(B{1,1}(i,1),B{1,1}(i,2))=1;
% end
% 
% figure
% imshow(I)
% figure
% imshow(newB)

load gong.mat;
soundsc(y);