%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILENAME: Linking function               
% AUTHOR: Ya, tanglab@stanford   
% DATE: 9/15/2015
% Note: Linking part: add 2nd level perimeter matching 
%          Add entranceX to avoid extra countings of small drops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [record,minDistance,minDelPeri,minEntry] = level_10_5_linkDrops_extrema(record,max_distance,max_delPeri,max_entranceX,numframes)

maxIdx = zeros(1,numframes-1);  % 1/13/2015 
minDistance = cell(1,numframes-1);  % 3/20/2015 Record the distance travelled between two frames
minDelPeri = cell(1,numframes-1);   % 9/13/2015 Record perimeter change of the same drop between two frames
minEntry = cell(1,numframes-1);    % 9/15/2015 Record entrance x coordinates of just-entering drops centroid
newlabel=1000;

for i = 1:numframes-1
    try
        
        %% Current frame i %%
        count = 0; % 'count' is to prevent assign the same id to drops that enter the frame simultaneously
        id_cur = record{i}(:,1);
        x_cur = record{i}(:,2);
        y_cur = record{i}(:,3);
        peri_cur = record{i}(:,5);
        maxIdx(i) = max(id_cur); % 1/13/2015 refer to notebook for debug situation
        
        %% Next frame i+1 %%
        id_next = record{i+1}(:,1);
        x_next = record{i+1}(:,2);
        y_next = record{i+1}(:,3);
        peri_next = record{i+1}(:,5);
        n_next = length(id_next);
        
        
        
        %% Centroid displacement match %%
        for j = 1:n_next
            distance = sqrt((x_next(j)-x_cur).^2+(y_next(j)-y_cur).^2); % 'distance' is a column vector, it compares the distances of one droplet in next frame to all droplets in current frame
            index_stay = find(distance < max_distance); % 'index_stay' gives the row index of drop j in record{i}.
            % In most cases, 'index_stay' has either one element or zero element. But
            % it could have more than one element if 'max_distance' is not proper chosen
            minDistance{i}(j) = min(distance);
            
            %  Case I: empty centroid match (this is a new drop)
            if isempty(index_stay)
                minEntry{i}(j) = x_next(j);
                if x_next(j) > max_entranceX
                    %                 id_next(j) = nan;   % Subcase I: this is a small drop (size comparable to area_floor) already in the field of view, disappear and appear again
                    %                 newlabel=newlabel+1;
                    count = count +1;
                    id_next(j) = max( maxIdx ) + count;
                else
                    count = count +1;
                    id_next(j) = max( maxIdx ) + count;
                end
                
                %  Case II: successful centroid match (same drop centroid displacement smaller than max_distance)
            elseif length(index_stay)==1
                minDelPeri{i}(j) = abs( peri_next(j) - peri_cur(index_stay) );
                if abs( peri_next(j) - peri_cur(index_stay) ) < max_delPeri    % Perimeter match: the perimeter of a drop between two frames barely varies (9/13/2015)
                    id_next(j) = id_cur(index_stay);
                else    % If the perimeters in two frames doesn't match, it is a small-size new drop (9/13/2015)
                    id_next(j) = id_cur(index_stay);
                    %                 count = count +1;
                    %                 id_next(j) = max( maxIdx ) + count;
                end
                
                %  Case III: two or more 'distance' elements are smaller than 'max_distance'
            else
                [~,index_stay] = min(distance);
                minDelPeri{i}(j) = abs( peri_next(j) - peri_cur(index_stay) );
                if abs( peri_next(j) - peri_cur(index_stay) ) < max_delPeri     % Perimeter match
                    id_next(j) = id_cur(index_stay);
                else  % Failed perimeter match
                    count = count +1;
                    id_next(j) = max( maxIdx ) + count;
                end
            end
        end
        record{i+1}(:,1) = id_next;
        
    end
end

end

