
for i = 219:342
    try
        filename = sprintf('7days1/test/%d.txt', i);
        T = readtable(filename);
    catch
        continue
    end
    T = table2array(T);
end

figure;
T1 = T(1:11, :);
T2 = T(12:131, :);
x = T1(:, 3);
y = T1(:,4);
z = T1(:, 5);
plot3(x, y, z, '-r', 'LineWidth', 1, 'MarkerSize', 2);
hold on
plot3(T2(:, 3), T2(:, 4), T2(:, 5), 'ob-', 'LineWidth', 1, 'MarkerSize', 8);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Future Trajectory')
lgd = legend('Past Trajectory', 'Predicted Future Trajectory Distribution', 'Location','best');
hold off
xlim([0, 5])
ylim([-0.3, 1])
zlim([0, 1])
print('future trajectory', '-dpng', '-r300');
