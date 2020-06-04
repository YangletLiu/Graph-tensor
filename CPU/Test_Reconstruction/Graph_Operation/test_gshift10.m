fid = fopen('result_gshift.txt','a+');
for t=1:10
    tic;
%     k = t*1000;
%     sparsity=0.4;
%     A=zeros(k,k);
%     for i=1:k
%         for j=(i+1):k
%             if rand<sparsity
%                 temp=rand;
%                 A(i,j)=temp;
%                 A(j,i)=temp;
%             end
%         end
%     end
    A=load(strcat('/home/kanwang/data/sensor',num2str(t*100),'.txt'));
    [k,~]=size(A);
    m = 1000;
    n = 1000;
    f = rand(m,n,k);
    F = zeros(m,n,k);
    clear i;
    clear j;
for i=1:k
    for j=1:k 
        F(:, :, i) = F(:, :, i) + A(i, j)*f(:, :, j);
    end
end 
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes: %d runtime: %f \n',k,toc);  
end
fclose(fid);