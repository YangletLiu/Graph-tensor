fid=fopen('result_igft.txt','a+');
for i=1:10
    tic;
    L=load(strcat('/home/kanwang/data/sensor',num2str(i*1000),'.txt'));
    [U, S] = eig(L);
    [k,~]=size(L);
    U=U';
    m = 10000;
    n = 1;
    f = rand(m,n,k);
    F = inverse_graph_fourier_transform(f, U); 
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes:%d time:%f \n',k,toc);
end
fclose(fid);
