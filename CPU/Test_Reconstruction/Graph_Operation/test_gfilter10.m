clear;
fid = fopen('result_gfilter.txt','a+');
for t=1:10
     tic;
     L=load(strcat('/home/kanwang/data/sensor',num2str(t*1000),'.txt'));
     [U, E] = eig(L);
     U=U';
     [k,~]=size(L);
%    k = t*1000;
%    sparsity=0.4;
%    A=zeros(k,k);
%    for i=1:k
%       for j=(i+1):k
%            if rand<sparsity
%                temp=rand;
%                A(i,j)=temp;
%                A(j,i)=temp;
%            end
%        end
%    end
%    [U,E] = graph_fourier_transform_matrix(A);
    m = 10000;
    n = 1;

    E=diag(E);
    lmax=max(E);
    g = @(x) sin(pi/4*x*(2/lmax));%regular 后面有定义
    s = rand(m,n,k);
    shat = graph_fourier_transform(s, U); 

    fd=zeros(length(E),1);%滤波器regular！！！！！
    for p=1:k
        fd(p,1)=g(E(p));
    end
    chat=zeros(m,n,k);
    for i=1:m
        for j=1:n
            %chat(i,j,:) = bsxfun(@times, conj(fd), shat(i,j,:));
            chat(i,j,:) = fd.*reshape(shat(i,j,:),k,1);
        end
    end
    c = inverse_graph_fourier_transform(chat, U); 
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes: %d  runtime: %f \n',k,toc); 
end
fclose(fid);
% function y = regular(val,d)
% 
% if d==0
%     y = sin(pi/4*val);
% else
%     output = sin(pi*(val-1)/2);
%     for k=2:d
%         output = sin(pi*output/2);
%     end
%     y = sin(pi/4*(1+output));
% end
% 
% end
