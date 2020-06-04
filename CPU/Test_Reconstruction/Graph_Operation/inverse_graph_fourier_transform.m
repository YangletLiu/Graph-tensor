%% 
%   Title              inverse graph fourier transform
%   Author         Xiaochen Han           
%   Date             Sep 23th, 2019
%   Version        1.0
%   Contact        guillermo_han97@sjtu.edu.cn
%
function [timeGraph] = inverse_graph_fourier_transform(fourierGraph, U)

% get inverse graph fourier transform matrix
%iU = pinv(graph_fourier_transform_matrix(graph.adjacent));
% iU = pinv(U);
iU = U';

% inverse graph fourier transform
[~, ~, n3] = size(fourierGraph);
timeGraph = zeros(size(fourierGraph));
for i=1:n3
    for j=1:n3 
        timeGraph(:, :, i) = timeGraph(:, :, i) + iU(i, j)*fourierGraph(:, :, j);
    end
end    
end