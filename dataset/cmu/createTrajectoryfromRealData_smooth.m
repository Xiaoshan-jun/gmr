%fileID = fopen('datasets/real/train/train.txt','w');
%fileID = fopen('datasets/real/val/val.txt','w');
%fileID = fopen('datasets/real/vis/vis.txt','w');
%fileID = fopen('plot.txt','w');
agentcount = 0;
t = 0;
b = 0.1;
fileID = fopen('train.txt','w');
smoothparameter = 0.001;
past = 20;
prediction = 20;
sample = 10;
%fileID = fopen('val.txt','w');
for i = 1:3089
    try
        filename = sprintf('train/%d.txt', i);
        T = readtable(filename);
    catch
        continue
    end
    T = table2array(T);
    %for each time, average the position
    a = size(T);
    tn = a(1);
    tn = floor(tn);
    half = tn/2;
    count = 0;
    sumx = 0;
    sumy = 0;
    sumz = 0;
    try
        t = T(1,1);
    catch
        delete(filename);
        continue
    end
    first = 1;
    fullhistoryx = [];
    fullhistoryy = [];
	fullhistoryz = [];
    %create fullhistory
    for k = 1:tn
        x = T(k, 3);
        y = T(k, 4);
        z = T(k,5);
        fullhistoryx(length(fullhistoryx)+1) = x;
        fullhistoryy(length(fullhistoryy)+1) = y;
        fullhistoryz(length(fullhistoryz)+1) = z;
    end
    %smooth full history
    l = sample;
    newfullhistoryx = [];
    newfullhistoryy = [];
    newfullhistoryz = [];
    for l = 2:length(fullhistoryx)
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > smoothparameter || abs(fullhistoryy(l) - fullhistoryy(l-1)) > smoothparameter || abs(fullhistoryx(l) - fullhistoryx(l-1)) > smoothparameter
            newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
            newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
            newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
        end
    end
    fullhistoryx = newfullhistoryx;
    fullhistoryy = newfullhistoryy;
    fullhistoryz = newfullhistoryz;

    %pick 10 history from full history
    for l = 1:10
        historyx = [];
        historyy = [];
        historyz = [];
        
    % Generate random integers between lower_bound and upper_bound
        random_values = randi([1, round(length(fullhistoryx)/2)], [1, past]);

        % Sort the random values in ascending order
        sorted_values = sort(random_values);
        %gt
%         filename = sprintf('val/testgt%d-%d.txt', i,l);
%         fileID = fopen(filename,'w');
        for t = 1:past
            if t == 1
                h = 'new';
            elseif t <= past
                h = 'past';
            else
                h = 'future';
            end
            historyx(t) = fullhistoryx(sorted_values(t));
            historyy(t) = fullhistoryy(sorted_values(t));
            historyz(t) = fullhistoryz(sorted_values(t));
           fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, historyx(t),historyy(t),historyz(t));
        end
        random_values = randi([round(length(fullhistoryx)/2), length(fullhistoryx)], [1, prediction]);
        sorted_values = sort(random_values);
        for t = past+1 : past + prediction
            if t == 1
                h = 'new';
            elseif t <= past
                h = 'past';
            else
                h = 'future';
            end
            historyx(t) = fullhistoryx(sorted_values(t - past));
            historyy(t) = fullhistoryy(sorted_values(t - past));
            historyz(t) = fullhistoryz(sorted_values(t - past));
           fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, historyx(t),historyy(t),historyz(t));
        end
        %disrupt
%         for d = 1:10
% %             filename = sprintf('val/testdisrupt%d-%d-%d.txt', i,l,d);
% %             fileID = fopen(filename,'w');
%             for t = 1:20
%                 if t == 1
%                     h = 'new';
%                 elseif t <= 10
%                     h = 'past';
%                 else
%                     h = 'future';
%                 end
%                newhistoryx = historyx(t);
%                newhistoryy = historyy(t);
%                newhistoryz = historyz(t);
%                dx = 0.1*historyx(t)*(rand()-0.5);
%                dy = 0.1*historyy(t)*(rand()-0.5);
%                dz = 0.1*historyz(t)*(rand()-0.5);
%                newhistoryx = newhistoryx + dx;
%                newhistoryy = newhistoryy + dy;
%                newhistoryz = newhistoryz + dz;
%                fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, newhistoryx,newhistoryy,newhistoryz);
%             end
%         end
    end

end
% str = sprintf('All Training Trajectory(Smooth Real Data).png');
% print(gcf,str,'-dpng','-r900'); 
fclose(fileID);

