%linear
%initial position
initial_Position1 = 0;
initial_Position2 = 1;
CT = 10; %collision thereshold
%------------------------do not change code below----------------------
if mod(initial_Position1,8) == 0
state1 = [400, 400, 140];
elseif mod(initial_Position1,8) == 1
state1 = [-400, 400, 140];
elseif mod(initial_Position1,8) == 2
state1 = [400, -400, 140];
elseif mod(initial_Position1,8)== 3
state1 = [200, 400, 140];
elseif mod(initial_Position1,8) == 4
state1 = [400, 200, 140 ];
elseif mod(initial_Position1,8) == 5
state1 = [200, -400, 140];
elseif mod(initial_Position1,8) == 6
state1 = [-400, 200, 140];
elseif mod(initial_Position1,8) == 7
state1 = [-400, -400, 140 ];
end
if mod(initial_Position2,8) == 0
state2 = [400, 400, 140];
elseif mod(initial_Position2,8) == 1
state2 = [-400, 400, 140];
elseif mod(initial_Position2,8) == 2
state2 = [400, -400, 140];
elseif mod(initial_Position2,8)== 3
state2 = [200, 400, 140];
elseif mod(initial_Position2,8) == 4
state2 = [400, 200, 140 ];
elseif mod(initial_Position2,8) == 5
state2 = [200, -400, 140];
elseif mod(initial_Position2,8) == 6
state2 = [-400, 200, 140];
elseif mod(initial_Position2,8) == 7
state2 = [-400, -400, 140 ];
end


fileID = fopen('agent1.txt','w');
fileID2 = fopen('agent2.txt','w');
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
simulationtime = 100000;
destination = [0, 0, 0];
first = 1;        
for t = 1: 10
    vb = (destination - state1)/(21 - t);
    xv = vb(1);
    yv = vb(2);
    zv = vb(3);

    state1(1) = state1(1) + xv;
    state1(2) = state1(2) + yv;
    state1(3) = max(state1(3) + zv,0); % bound z above zero
    if first == 1
        h = 'new';
        first = 0;
    elseif t <= 10
        h = 'past';
    else
        h = 'future';
    end
    fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, state1(1),state1(2),state1(3));
    vb = (destination - state2)/(21 - t);
    xv = vb(1);
    yv = vb(2);
    zv = vb(3);

    state2(1) = state2(1) + xv;
    state2(2) = state2(2) + yv;
    state2(3) = max(state2(3) + zv,0); % bound z above zero
    fprintf(fileID2,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, state2(1),state2(2),state2(3));
end
initialstate1 = state1;
initialstate2 = state2;
for i = 0 : 100000 %trajectory number
    state1 = initialstate1;
    state2 = initialstate2;
    for t = 11 : 20
        vb = (destination - state1)/(21 - t);
        xv = vb(1) + 25*(rand() - 0.5);
        yv = vb(2) + 25*(rand() - 0.5);
        zv = vb(3) + 9*(rand() - 0.5);

        state1(1) = state1(1) + xv;
        state1(2) = state1(2) + yv;
        state1(3) = max(state1(3) + zv,0); % bound z above zero

        vb = (destination - state2)/(21 - t);
        xv = vb(1) + 25*(rand() - 0.5);
        yv = vb(2) + 25*(rand() - 0.5);
        zv = vb(3) + 9*(rand() - 0.5);

        state2(1) = state2(1) + xv;
        state2(2) = state2(2) + yv;
        state2(3) = max(state2(3) + zv,0); % bound z above zero
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
ans = [t1/simulationtime,t2/simulationtime,t3/simulationtime,t4/simulationtime,t5/simulationtime,t6/simulationtime,t7/simulationtime,t8/simulationtime,t9/simulationtime,t10/simulationtime];
save('ans.mat', 'ans');
%str = sprintf('linear%d.png', i);

