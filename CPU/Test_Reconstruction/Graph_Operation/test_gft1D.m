tic;
% L = load('~/data/sensor10000.txt');
% G = gsp_sensor(10000000);
% [U, S] = eig(full(L));
% U=U';
% [k,~]=size(L);
k = 20000;
sparsity=0.4;
A=zeros(k,k);
for i=1:k
    for j=(i+1):k
        if rand<sparsity
            temp=rand;
            A(i,j)=temp;
            A(j,i)=temp;
        end
    end
end
U = graph_fourier_transform_matrix(A);
m = 1;
n = 1;
f = rand(m,n,k);
F = graph_fourier_transform(f, U); 
toc;
disp(['nodes:',num2str(k)]);
disp(['runtime: ',num2str(toc)]);
