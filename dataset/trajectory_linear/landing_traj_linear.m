%%
%straight line
fileID = fopen('train.txt','w');
%fileID = fopen('val.txt','w');



for i = 0 : 100000 %trajectory number
    if mod(i,8) == 0
    state = [400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 1
    state = [-400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 2
    state = [400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8)== 3
    state = [200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 4
    state = [400 + 200*(rand()-0.5) , 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 5
    state = [200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 6
    state = [-400 + 200*(rand()-0.5) , 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,8) == 7
    state = [-400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    end
    destination = [0, 0, 0];
    first = 1;        
%     filename = sprintf('val/testgt%d.txt', i);
%     fileID = fopen(filename,'w');
    for t = 1 : 20
        vb = (destination - state)/(21 - t);
        xv = vb(1) + 25*(rand() - 0.5);
        yv = vb(2) + 25*(rand() - 0.5);
        zv = vb(3) + 9*(rand() - 0.5);

        state(1) = state(1) + xv;
        state(2) = state(2) + yv;
        state(3) = max(state(3) + zv,0); % bound z above zero

        if first == 1
            h = 'new';
            first = 0;
        elseif t <= 10
            h = 'past';
        else
            h = 'future';
        end
        fprintf(fileID,'%s\t%4.2f\t%4.2f\t%4.2f\n',h, state(1),state(2),state(3));
        
    end
    
   
end
