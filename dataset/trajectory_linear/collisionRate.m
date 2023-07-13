%scenario 1
%starts (400,400,140) and (400,0,140)
%straight line
fileID = fopen('agent1.txt','w');
fileID2 = fopen('agent2.txt','w');
mvx = 25; %max horizontal speed
mvy = 25; %max horizontal speed
mvz = 9;  %max descend speed
CT = 20; %collision thereshold
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
for i = 0 : 100000 %trajectory number
    state1 = [400  , 400 , 140 ];
    state2 = [400  , 0, 140];
    destination = [0, 0, 0];
    first = 1;        
%     filename = sprintf('val/testgt%d.txt', i);
%     fileID = fopen(filename,'w');
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
        if i == 1
            fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, state1(1),state1(2),state1(3));
        end
        vb = (destination - state2)/(21 - t);
        xv = vb(1);
        yv = vb(2);
        zv = vb(3);        

        state2(1) = state2(1) + xv;
        state2(2) = state2(2) + yv;
        state2(3) = max(state2(3) + zv,0); % bound z above zero
        if i == 1
        fprintf(fileID2,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, state2(1),state2(2),state2(3));
        end
    end
    for t = 11 : 20
        vb = (destination - state1)/(21 - t);
        xv = vb(1) + 15*(rand() - 0.5);
        yv = vb(2) + 15*(rand() - 0.5);
        zv = vb(3) + 9*(rand() - 0.5);

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
ans = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10];
save('ans.mat', 'ans');
%str = sprintf('linear%d.png', i);

