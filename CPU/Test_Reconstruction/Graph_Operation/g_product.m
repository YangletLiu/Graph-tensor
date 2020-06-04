%% 
%   Title              graph product  
%   Author         Xiaochen Han           
%   Date             Sep 23th, 2019
%   Version        1.0
%   Contact        guillermo_han97@sjtu.edu.cn
%
function [timeG3] = g_product(timeG1, timeG2, U)
% g1 * g2 = L^{-1}(L(g1)*L(g2))

[n1, n2, n3] = size(timeG1);
[n2, n4, n3] = size(timeG2);
fourierG1 = graph_fourier_transform(timeG1, U);
fourierG2 = graph_fourier_transform(timeG2, U);

fourierG3 = zeros(n1, n4, n3);
for i=1:n3
    fourierG3(:, :, i) = fourierG1(:, :, i)*fourierG2(:, :, i);
end
timeG3 = inverse_graph_fourier_transform(fourierG3, U);

end