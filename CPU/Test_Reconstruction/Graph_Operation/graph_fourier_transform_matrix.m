%%
%   Title              get graph fourier transform matrix
%   Author         Xiaochen Han           
%   Date             Sep 23th, 2019
%   Version        1.0
%   Contact        guillermo_han97@sjtu.edu.cn
%

function [U,E] = graph_fourier_transform_matrix(W)

% calculate D
[nNode, ~] = size(W);

D = zeros(nNode, nNode);
for iNode=1:nNode
    D(iNode, iNode) = sum(W(iNode, :));
end

% calculate D1 = D^{-1/2}
D1 = zeros(nNode, nNode);
for iNode=1:nNode
    if D(iNode, iNode) ~= 0
        D1(iNode, iNode) = D(iNode, iNode).^(-1/2);
    end
end

% calculate L
I = diag(ones(nNode, 1));
L = I - D1*W*D1;
%L = D - W;
% L = zeros(nNode, nNode);
% for i=1:nNode
%     for j=1:nNode
%         if i==j && D(i, i) ~= 0
%             L(i, j) = 1;
%         elseif i~=j && W(i, j) ~= 0
%             L(i, j) = -1/sqrt(D(i, i)*D(j, j));
%         end
%     end
% end
        
% calaulte graph fourier transform matrix U and corresponding inverse iU
[eigVec, eigVal] = eig(L);
U = eigVec';
%U = flipud(eigVec');

% verify
E = eigVal;
%fprintf("Verify UL=EU:  max(UL-EU)=%.2f\n", max(max(U*L-E*U)));

end
