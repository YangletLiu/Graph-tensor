%% 
%   Title              graph SVD
%   Author         Xiaochen Han           
%   Date             Sep 23th, 2019
%   Version        1.0
%   Contact        guillermo_han97@sjtu.edu.cn
%
function [U, S, V] = g_SVD(timeGraph, Uf)

[n1, n2, n3] = size(timeGraph);
fourierGraph = graph_fourier_transform(timeGraph, Uf);

% U: n1xn1xn3    S: n1xn2xn3    V: n2xn2xn3
fourierU = zeros(n1, n1, n3);
fourierS = zeros(n1, n2, n3);
fourierV = zeros(n2, n2, n3);

for i=1:n3
    [fourierU(:, :, i), fourierS(:, :, i), fourierV(:, :, i)] = svd(fourierGraph(:, :, i));
end

U = inverse_graph_fourier_transform(fourierU, Uf);
S = inverse_graph_fourier_transform(fourierS, Uf);
V = inverse_graph_fourier_transform(fourierV, Uf);

end