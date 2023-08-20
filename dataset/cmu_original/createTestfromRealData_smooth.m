%fileID = fopen('datasets/real/train/train.txt','w');
%fileID = fopen('datasets/real/val/val.txt','w');
%fileID = fopen('datasets/real/vis/vis.txt','w');
%fileID = fopen('plot.txt','w');
agentcount = 0;
t = 0;
b = 0.1;
smoothparameter = 0.001;
past = 11;
prediction = 240;
skip = 1;
for i = 1:3089
    try
        filename = sprintf('7days1/test/%d.txt', i);
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
        continue
    end
    first = 1;
    fullhistoryx = [];
    fullhistoryy = [];
	fullhistoryz = [];
    fullhistorywx = [];
    fullhistorywy = [];
    %create fullhistory
    for k = 1:tn
        x = T(k, 3);
        y = T(k, 4);
        z = T(k,5);
        wx = T(k, 6);
        wy = T(k, 7);
        fullhistoryx(length(fullhistoryx)+1) = x;
        fullhistoryy(length(fullhistoryy)+1) = y;
        fullhistoryz(length(fullhistoryz)+1) = z;
        fullhistorywx(length(fullhistorywx)+1) = wx;
        fullhistorywy(length(fullhistorywy)+1) = wy;
    end
    %smooth full history
%     newfullhistoryx = [];
%     newfullhistoryy = [];
%     newfullhistoryz = [];
%     for l = 2:length(fullhistoryx)
%         if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > smoothparameter || abs(fullhistoryy(l) - fullhistoryy(l-1)) > smoothparameter || abs(fullhistoryx(l) - fullhistoryx(l-1)) > smoothparameter
%             newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
%             newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
%             newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
%         end
%     end
%     fullhistoryx = newfullhistoryx;
%     fullhistoryy = newfullhistoryy;
%     fullhistoryz = newfullhistoryz;
%     if size(fullhistoryx) == 0
%         continue
%     end
    lastframe = length(fullhistoryx) - past - prediction * skip;
    %pick 10 history from full history
    for l = 1:100:lastframe
        filename = sprintf('7days1/val/testgt%d.txt', i*10000 + l);
        fileID = fopen(filename,'w');
        historyx = [];
        historyy = [];
        historyz = [];
        
        for t = 0:past-1
            if t == 0
                h = 'new';
                fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\n',h, fullhistoryx(l + t),fullhistoryy(l + t),fullhistoryz(l + t), fullhistorywx(1 + t), fullhistorywy(1 + t));
                continue
            elseif t <= past-1
                h = 'past';
            else
                h = 'future';
            end
           fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, fullhistoryx(l + t),fullhistoryy(l + t),fullhistoryz(l + t));
        end
        for t = past -1 + skip : skip : past -1 + prediction*skip
            if t == 0
                h = 'new';
            elseif t <= past - 1
                h = 'past';
            else
                h = 'future';
            end
           fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, fullhistoryx(l + t), fullhistoryy(l + t), fullhistoryz(l + t));
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
    fclose(fileID);
    end

end
% str = sprintf('All Training Trajectory(Smooth Real Data).png');
% print(gcf,str,'-dpng','-r900'); 

