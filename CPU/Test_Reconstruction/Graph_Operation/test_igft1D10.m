fid = fopen('result.txt','a+');
for t=10:20
    tic;
    k = t*1000;
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
    F = inverse_graph_fourier_transform(f, U); 
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes:%f time:%f \n',k,toc);
end
