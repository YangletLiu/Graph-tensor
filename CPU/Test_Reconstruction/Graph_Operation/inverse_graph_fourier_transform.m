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
%%
% for i=1:n3
%     for j=1:n3 
%         timeGraph(:, :, i) = timeGraph(:, :, i) + iU(i, j)*fourierGraph(:, :, j);
%     end
% end  

%%
% for i=1:n2
%     timeGraph(:,i,:)=(iU*(reshape(fourierGraph(:,i,:),n1,n3))')';
% end

%%
for i=1:n1
   for j=1:n2
       timeGraph(i, j, :) = iU*reshape(fourierGraph(i, j, :),[],1);
   end
end

end
