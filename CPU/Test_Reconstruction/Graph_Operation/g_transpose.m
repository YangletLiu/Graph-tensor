%% 
%   Title              graph transpose
%   Author         Xiaochen Han           
%   Date             Sep 23th, 2019
%   Version        1.0
%   Contact        guillermo_han97@sjtu.edu.cn
%
function [timeGT] = g_transpose(timeG, U)

[n1, n2, n3] = size(timeG);
fourierGT = zeros(n2, n1, n3);

% transpose each frontal slice in graph fourier domain
fourierG  = graph_fourier_transform(timeG, U);
for i=1:n3
    fourierGT(:, :, i) = fourierG(:, :, i)';
end

% get time domain graph
timeGT = inverse_graph_fourier_transform(fourierGT, U);
end