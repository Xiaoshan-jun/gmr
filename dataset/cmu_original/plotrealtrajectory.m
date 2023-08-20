
for i = 0:2
    try
        filename = sprintf('7days1/test/%d.txt', i);
        T = readtable(filename);
    catch
        continue
    end
    all = table2array(T);
end

for i = 2:5000
    try
        filename = sprintf('7days1/test/%d.txt', i);
        T = readtable(filename);
    catch
        continue
    end
    T = table2array(T);
    all = [all; T];
end
sortedall = sortrows(all, 1);
%     plot3(T(:, 3), T(:, 4), T(:, 5), 'b-', 'LineWidth', 2);
%     hold on
%     xlabel('X-axis');
%     ylabel('Y-axis');
%     zlabel('Z-axis');
%     xlim([-5, 5])
%     ylim([-5, 5])
%     zlim([-5, 5])
