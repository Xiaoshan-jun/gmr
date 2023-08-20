%scenario 1
clc
clear all
%straight line
p1 = 5;
p2 = 7;
CT = 0.1; %collision thereshold
past = 20;
prediction = 20;
%----------------------------------------
fileID = fopen('agent1.txt','w');
fileID2 = fopen('agent2.txt','w');
smoothparameter = 0.003;
mvx = 25; %max horizontal speed
mvy = 25; %max horizontal speed
mvz = 9;  %max descend speed
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
t11 = 0;
t12 = 0;
t13 = 0;
t14 = 0;
t15 = 0;
t16 = 0;
t17 = 0;
t18 = 0;
t19 = 0;
t20 = 0;
filename = sprintf('test/%d.txt', p1);
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
        x = T(k, 3);
        y = T(k, 4);
        z = T(k,5);
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
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > smoothparameter || abs(fullhistoryy(l) - fullhistoryy(l-1)) > smoothparameter || abs(fullhistoryx(l) - fullhistoryx(l-1)) > smoothparameter
            newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
            newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
            newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
        end
    end
    fullhistoryx1 = newfullhistoryx;
    fullhistoryy1 = newfullhistoryy;
    fullhistoryz1 = newfullhistoryz;
    filename2 = sprintf('test/%d.txt', p2);
    T = readtable(filename2);
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
        x = T(k, 3);
        y = T(k, 4);
        z = T(k,5);
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
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > smoothparameter || abs(fullhistoryy(l) - fullhistoryy(l-1)) > smoothparameter || abs(fullhistoryx(l) - fullhistoryx(l-1)) > smoothparameter
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
        random_values = randi([1, round(length(fullhistoryx1)/2)], [1, past]);

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
        random_values = randi([1, round(length(fullhistoryx2)/2)], [1, past]);

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
            historyx(t) = fullhistoryx2(sorted_values(t));
            historyy(t) = fullhistoryy2(sorted_values(t));
            historyz(t) = fullhistoryz2(sorted_values(t));
           fprintf(fileID2,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, historyx(t),historyy(t),historyz(t));
        end
end
%%
simulationtime = 100000;
for i = 0 : simulationtime %trajectory number
    state1 = [0 ,0 ,0];
    state2 = [0 ,0 ,0];
    random_values = randi([round(length(fullhistoryx1)/2), length(fullhistoryx1)], [1, prediction]);
    sorted_values1 = sort(random_values);
    random_values = randi([round(length(fullhistoryx2)/2), length(fullhistoryx2)], [1, prediction]);
    sorted_values2 = sort(random_values);
    for t = past+1 : past + prediction
           state1(1) = fullhistoryx1(sorted_values1(t-past));
           state1(2) = fullhistoryy1(sorted_values1(t-past));
           state1(3) = fullhistoryz1(sorted_values1(t-past));
           state2(1) = fullhistoryx2(sorted_values2(t-past));
           state2(2) = fullhistoryy2(sorted_values2(t-past));
           state2(3) = fullhistoryz2(sorted_values2(t-past));
%            dx = 0.1*state1(1)*(rand()-0.5);
%            dy = 0.1*state1(2)*(rand()-0.5);
%            dz = 0.1*state1(3)*(rand()-0.5);
%            state1(1) = state1(1) + dx;
%            state1(2) = state1(2) + dy;
%            state1(3) = state1(3) + dz;
%            dx = 0.1*state2(1)*(rand()-0.5);
%            dy = 0.1*state2(2)*(rand()-0.5);
%            dz = 0.1*state2(3)*(rand()-0.5);
%            state2(1) = state2(1) + dx;
%            state2(2) = state2(2) + dy;
%            state2(3) = state2(3) + dz;
        if t == past + 1 && pdist2(state1, state2) < CT
            t1 = t1 + 1;
        end
        if t == past + 2 && pdist2(state1, state2) < CT
            t2 = t2 + 1;
        end
        if t == past + 3 && pdist2(state1, state2) < CT
            t3 = t3 + 1;
        end
        if t == past + 4 && pdist2(state1, state2) < CT
            t4 = t4 + 1;
        end
        if t == past + 5 && pdist2(state1, state2) < CT
            t5 = t5 + 1;
        end
        if t == past + 6 && pdist2(state1, state2) < CT
            t6 = t6 + 1;
        end
        if t == past + 7 && pdist2(state1, state2) < CT
            t7 = t7 + 1;
        end
        if t == past + 8 && pdist2(state1, state2) < CT
            t8 = t8 + 1;
        end
        if t == past + 9 && pdist2(state1, state2) < CT
            t9 = t9 + 1;
        end
        if t == past + 10 && pdist2(state1, state2) < CT
            t10 = t10 + 1;
        end
        if t == past + 11 && pdist2(state1, state2) < CT
            t11 = t11 + 1;
        end
        if t == past + 12 && pdist2(state1, state2) < CT
            t12 = t12 + 1;
        end
        if t == past + 13 && pdist2(state1, state2) < CT
            t13 = t13 + 1;
        end
        if t == past + 14 && pdist2(state1, state2) < CT
            t14 = t14 + 1;
        end
        if t == past + 15 && pdist2(state1, state2) < CT
            t15 = t15 + 1;
        end
        if t == past + 16 && pdist2(state1, state2) < CT
            t16 = t16 + 1;
        end
        if t == past + 17 && pdist2(state1, state2) < CT
            t17 = t17 + 1;
        end
        if t == past + 18 && pdist2(state1, state2) < CT
            t18 = t18 + 1;
        end
        if t == past + 19 && pdist2(state1, state2) < CT
            t19 = t19 + 1;
        end
        if t == past + 20 && pdist2(state1, state2) < CT
            t20 = t20 + 1;
        end
            
    end
end
ans = [t1/simulationtime,t2/simulationtime,t3/simulationtime,t4/simulationtime,t5/simulationtime,t6/simulationtime,t7/simulationtime,t8/simulationtime,t9/simulationtime,t10/simulationtime, t11/simulationtime,t12/simulationtime,t13/simulationtime,t14/simulationtime,t15/simulationtime,t16/simulationtime,t17/simulationtime,t18/simulationtime,t19/simulationtime,t20/simulationtime];
save('ans.mat', 'ans');
%str = sprintf('linear%d.png', i);

