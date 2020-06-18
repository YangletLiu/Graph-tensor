clear all
close all

%% ====================== Load data ==============================
addpath('Tubal_Alt_Min', 'Graph_Operation') ;

load CPark360
T = T(:, :, 1:20);
[m,n,k]=size(T); 

% adjacent matrix
adjacent = zeros(k, k);
adjacent(1, k) = 1;
adjacent(k, 1) = 1;
for i=1:k-1
    adjacent(i, i+1) = 1;
    adjacent(i+1, i) = 1;
end
U = graph_fourier_transform_matrix(adjacent);

%[T1,R]=de_noise(T, U); 
%r=R; 
T1=T;
r=20;

    num_miss_frame = 10;
    omega = ones(m,n,k);
    miss_index = [3 5 6 8 9 12 14 15 17 18 4 16];      
    for i=1:num_miss_frame
        omega(:,:,miss_index(i)) = 0;
    end
    clear i;

    T_omega = omega .* T1;
    T_omega_f = graph_fourier_transform(T_omega, U);     
    omega_f = graph_fourier_transform(omega, U);            
    %T_omega_f = fft(T_omega,[],3);
    %omega_f = fft(omega, [], 3);
% X: m * r * k
% Y: r * n * k
%% Given Y, do LS to get X
    Y = rand(r, n, k);
    
    Y_f = graph_fourier_transform(Y, U);
    %Y_f = fft(Y, [], 3);

% do the transpose for each frontal slice
    Y_f_trans = zeros(n,r,k);
    X_f = zeros(m,r,k);
    T_omega_f_trans = zeros(n,m,k);
    omega_f_trans = zeros(n,m,k);
for i = 1: k
     Y_f_trans(:,:,i) = Y_f(:,:,i)';
     T_omega_f_trans(:,:,i) = T_omega_f(:,:,i)';
     omega_f_trans(:,:,i) = omega_f(:,:,i)';
end
tic;
iter=1;
while iter <=10
    fprintf('Sample--%f---Round--#%d\n', 2, iter);
    [X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans * 1/k, Y_f_trans);
    %[X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans, Y_f_trans);
    
    for i =1:k
        X_f(:,:,i) = X_f_trans(:,:,i)';
    end

    % Given X, do LS to get Y
    [Y_f] = alter_min_LS_one_step(T_omega_f, omega_f * 1/k, X_f);
    %[Y_f] = alter_min_LS_one_step(T_omega_f, omega_f, X_f);
    
    for i = 1: k
        Y_f_trans(:,:,i) = Y_f(:,:,i)';
    end
    
    iter = iter + 1;
end

% The relative error:
X_est = inverse_graph_fourier_transform(X_f, U);         
Y_est = inverse_graph_fourier_transform(Y_f, U);         
T_est = g_product(X_est, Y_est, U);                                   
% X_est = ifft(X_f, [], 3); 
% Y_est = ifft(Y_f, [], 3);
% T_est = tprod(X_est, Y_est);
T_est1=abs(T_est);

% into the same scale
T1 = T1/255;
T_est1 = T_est1*mean(T1(:))/mean(T_est1(:));
toc;
temp = T1 - T_est1;   
error = norm(temp(:)) / norm(T1(:));
%PSNR = [];
%for i=1:num_miss_frame
%    PSNR = [PSNR, psnr(T1(:, :, miss_index(i)), T_est1(:, :, miss_index(i)))];
%end
fprintf("error=%.4f\n", error);
disp(['runtime:',num2str(toc)]);
%for i=1:k
%subplot(2,k,i);imagesc(T1(:,:,i));axis off;
%colormap(gray);
%subplot(2,k,i+k);imagesc(T_est1(:,:,i));axis off;
%colormap(gray);
%end

%profile viewer
