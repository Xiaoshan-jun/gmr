%scenario 1
%starts (400,400,140) and (400,0,140)
%straight line
fileID = fopen('agent1.txt','w');
fileID2 = fopen('agent2.txt','w');
mvx = 25; %max horizontal speed
mvy = 25; %max horizontal speed
mvz = 9;  %max descend speed
CT = 0.1; %collision thereshold
t1 = 0;
t2 = 0;
t3 = 0;
t4 = 0;
t5 = 0;
t6 = 0;
t7 = 0;
t8 = 0;
t9 = 0;
t10 = 0;
filename = 'realdata2/flightdata8.csv';
    T = readtable(filename);
    T = table2array(T);
    for k = 1:size(T)
        T(k, 1) = round(T(k, 1),1); %round all the time
    end
    %for each time, average the position
    a = size(T);
    tn = a(1);
    tn = floor(tn);
    half = tn/2;
    count = 0;
    sumx = 0;
    sumy = 0;
    sumz = 0;
    t = T(1,1);
    first = 1;
    fullhistoryx = [];
    fullhistoryy = [];
	fullhistoryz = [];
    %create fullhistory
    for k = 1:tn
        x = T(k, 2);
        y = T(k, 3);
        z = T(k,4);
        fullhistoryx(length(fullhistoryx)+1) = x;
        fullhistoryy(length(fullhistoryy)+1) = y;
        fullhistoryz(length(fullhistoryz)+1) = z;
    end
    %smooth full history
    l = 2;
    newfullhistoryx = [];
    newfullhistoryy = [];
    newfullhistoryz = [];
    for l = 2:length(fullhistoryx)
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > 0.001 || abs(fullhistoryy(l) - fullhistoryy(l-1)) > 0.001 || abs(fullhistoryx(l) - fullhistoryx(l-1)) > 0.001
            newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
            newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
            newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
        end
    end
    fullhistoryx1 = newfullhistoryx;
    fullhistoryy1 = newfullhistoryy;
    fullhistoryz1 = newfullhistoryz;
filename = 'realdata2/flightdata18.csv';
    T = readtable(filename);
    T = table2array(T);
    for k = 1:size(T)
        T(k, 1) = round(T(k, 1),1); %round all the time
    end
    %for each time, average the position
    a = size(T);
    tn = a(1);
    tn = floor(tn);
    half = tn/2;
    count = 0;
    sumx = 0;
    sumy = 0;
    sumz = 0;
    t = T(1,1);
    first = 1;
    fullhistoryx = [];
    fullhistoryy = [];
	fullhistoryz = [];
    %create fullhistory
    for k = 1:tn
        x = T(k, 2);
        y = T(k, 3);
        z = T(k,4);
        fullhistoryx(length(fullhistoryx)+1) = x;
        fullhistoryy(length(fullhistoryy)+1) = y;
        fullhistoryz(length(fullhistoryz)+1) = z;
    end
    %smooth full history
    l = 2;
    newfullhistoryx = [];
    newfullhistoryy = [];
    newfullhistoryz = [];
    for l = 2:length(fullhistoryx)
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > 0.001 || abs(fullhistoryy(l) - fullhistoryy(l-1)) > 0.001 || abs(fullhistoryx(l) - fullhistoryx(l-1)) > 0.001
            newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
            newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
            newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
        end
    end
    fullhistoryx2 = newfullhistoryx;
    fullhistoryy2 = newfullhistoryy;
    fullhistoryz2 = newfullhistoryz;
for l = 1:1
        historyx = [];
        historyy = [];
        historyz = [];
        
    % Generate random integers between lower_bound and upper_bound
        random_values = randi([1, length(fullhistoryx1)/2], [1, 10]);

        % Sort the random values in ascending order
        sorted_values = sort(random_values);
        %gt
%         filename = sprintf('val/testgt%d-%d.txt', i,l);
%         fileID = fopen(filename,'w');
        for t = 1:10
            if t == 1
                h = 'new';
            elseif t <= 10
                h = 'past';
            else
                h = 'future';
            end
            historyx(t) = fullhistoryx1(sorted_values(t));
            historyy(t) = fullhistoryy1(sorted_values(t));
            historyz(t) = fullhistoryz1(sorted_values(t));
           fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, historyx(t),historyy(t),historyz(t));
        end
end
for l = 1:1
        historyx = [];
        historyy = [];
        historyz = [];
        
    % Generate random integers between lower_bound and upper_bound
        random_values = randi([1, length(fullhistoryx2)/2], [1, 10]);

        % Sort the random values in ascending order
        sorted_values = sort(random_values);
        %gt
%         filename = sprintf('val/testgt%d-%d.txt', i,l);
%         fileID = fopen(filename,'w');
        for t = 1:10
            if t == 1
                h = 'new';
            elseif t <= 10
                h = 'past';
            else
                h = 'future';
            end
            historyx(t) = fullhistoryx2(sorted_values(t));
            historyy(t) = fullhistoryy2(sorted_values(t));
            historyz(t) = fullhistoryz2(sorted_values(t));
           fprintf(fileID2,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, historyx(t),historyy(t),historyz(t));
        end
end
%%
for i = 0 : 1000000 %trajectory number
    state1 = [0 ,0 ,0];
    state2 = [0 ,0 ,0];
    for t = 11 : 20
           random_values = randi([length(fullhistoryx1)/2, length(fullhistoryx1)], [1, 10]);
           sorted_values = sort(random_values);
           state1(1) = fullhistoryx1(sorted_values(t-10));
           state1(2) = fullhistoryy1(sorted_values(t-10));
           state1(3) = fullhistoryz1(sorted_values(t-10));
           random_values = randi([length(fullhistoryx2)/2, length(fullhistoryx2)], [1, 10]);
           sorted_values = sort(random_values);
           state2(1) = fullhistoryx2(sorted_values(t-10));
           state2(2) = fullhistoryy2(sorted_values(t-10));
           state2(3) = fullhistoryz2(sorted_values(t-10));
           dx = 0.1*state1(1)*(rand()-0.5);
           dy = 0.1*state1(2)*(rand()-0.5);
           dz = 0.1*state1(3)*(rand()-0.5);
           state1(1) = state1(1) + dx;
           state1(2) = state1(2) + dy;
           state1(3) = state1(3) + dz;
           dx = 0.1*state2(1)*(rand()-0.5);
           dy = 0.1*state2(2)*(rand()-0.5);
           dz = 0.1*state2(3)*(rand()-0.5);
           state2(1) = state2(1) + dx;
           state2(2) = state2(2) + dy;
           state2(3) = state2(3) + dz;
        if t == 11 && pdist2(state1, state2) < CT
            t1 = t1 + 1;
        end
        if t == 12 && pdist2(state1, state2) < CT
            t2 = t2 + 1;
        end
        if t == 13 && pdist2(state1, state2) < CT
            t3 = t3 + 1;
        end
        if t == 14 && pdist2(state1, state2) < CT
            t4 = t4 + 1;
        end
        if t == 15 && pdist2(state1, state2) < CT
            t5 = t5 + 1;
        end
        if t == 16 && pdist2(state1, state2) < CT
            t6 = t6 + 1;
        end
        if t == 17 && pdist2(state1, state2) < CT
            t7 = t7 + 1;
        end
        if t == 18 && pdist2(state1, state2) < CT
            t8 = t8 + 1;
        end
        if t == 19 && pdist2(state1, state2) < CT
            t9 = t9 + 1;
        end
        if t == 20 && pdist2(state1, state2) < CT
            t10 = t10 + 1;
        end
            
    end
end
ans = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10];
save('ans.mat', 'ans');
%str = sprintf('linear%d.png', i);

