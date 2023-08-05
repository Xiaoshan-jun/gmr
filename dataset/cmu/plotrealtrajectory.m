for i = 1:30
    try
        filename = sprintf('train/%d.txt', i);
        T = readtable(filename);
    catch
        continue
    end
    T = table2array(T);
    figure
    plot3(T(:, 3), T(:, 4), T(:, 5), 'b-', 'LineWidth', 2);
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    xlim([-5, 5])
    ylim([-5, 5])
    zlim([-5, 5])
end