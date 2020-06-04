fid = fopen('result.txt','a+');
for t=1:10
    tic;
    L=load(strcat('/home/kanwang/data/sensor',num2str(t*100),'.txt'));
    [U, E] = eig(L);
    U = U';
    [k,~] = size(L);
    m=1000;
    n=1000;
    s=rand(m,n,k);
    s_f = graph_fourier_transform(s, U);

    E = diag(E);
    lmax=max(E);
    H=zeros(k,k);
    for i=1:k
        alpha=rand;
        H(k,k)= alpha+alpha*E(k);
    end
    H_f = graph_fourier_transform(H, U);
    s_m = zeros(m,n,k);
    H_f=reshape(H_f,k,k);
    for j=1:n
        s_m(:,j,:)=(H_f*(reshape(s(:,j,:),m,k))')';
    end
    s_g = graph_fourier_transform(s_m, U);
             
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes: %d runtime: %f \n',k,toc);  
end
clear t;
for t=1:10
    tic;
    L=load(strcat('/home/kanwang/data/sensor',num2str(t*1000),'.txt'));
    [U, E] = eig(L);
    U = U';
    [k,~] = size(L);
    m=10000;
    n=1;
    s=rand(m,n,k);
    s_f = graph_fourier_transform(s, U);

    E = diag(E);
    lmax=max(E);
    H=zeros(k,k);
    for i=1:k
        alpha=rand;
        H(k,k)= alpha+alpha*E(k);
    end
    H_f = graph_fourier_transform(H, U);
    s_m = zeros(m,n,k);
    H_f=reshape(H_f,k,k);
    for j=1:n
        s_m(:,j,:)=(H_f*(reshape(s(:,j,:),m,k))')';
    end
    s_g = graph_fourier_transform(s_m, U);
             
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes: %d runtime: %f \n',k,toc);  
end
clear t;
for t=10:20
    tic;
    L=load(strcat('/home/kanwang/data/sensor',num2str(t*1000),'.txt'));
    [U, E] = eig(L);
    U = U';
    [k,~] = size(L);
    m=1;
    n=1;
    s=rand(m,n,k);
    s_f = graph_fourier_transform(s, U);

    E = diag(E);
    lmax=max(E);
    H=zeros(k,k);
    for i=1:k
        alpha=rand;
        H(k,k)= alpha+alpha*E(k);
    end
    H_f = graph_fourier_transform(H, U);
    s_m = zeros(m,n,k);
    H_f=reshape(H_f,k,k);
    for j=1:n
        s_m(:,j,:)=(H_f*(reshape(s(:,j,:),m,k))')';
    end
    s_g = graph_fourier_transform(s_m, U);
             
    toc;
    disp(['nodes:',num2str(k)]);
    disp(['runtime: ',num2str(toc)]);
    fprintf(fid,'nodes: %d runtime: %f \n',k,toc);  
end
fclose(fid);