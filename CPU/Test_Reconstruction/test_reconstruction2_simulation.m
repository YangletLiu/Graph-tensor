clear all
close all

%% ====================== Load data ==============================
addpath('Tubal_Alt_Min', 'Graph_Operation') ;
load walking
T = T(:, :, 1:5);
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


% simulation data generation
r = 50;
Uf = 3*rand(m, r, k);
S = zeros(r, r, k);
V = 3*rand(n, r, k);
for i=1:k
    S(:, :, i) = diag(sort(rand(r, 1), 'descend'));
end
S(:, 2:fix(r/2), :) = S(:, 2:fix(r/2), :)/10;
S(:, fix(r/2):end, :) = S(:, fix(r/2):end, :)/100;
T = g_product(g_product(Uf, S, U), g_transpose(V, U), U);


[m,n,k]=size(T); 
T1 = T;
r = min(m,n);

    num_miss_frame = 1;
    omega = ones(m, n, k);
    miss_index = [3];      
    for i=1:num_miss_frame
        omega(:,:,miss_index(i)) = 0;
    end
    clear i;
    
     T_omega = omega .* T1;
     T_omega_f = graph_fourier_transform(T_omega, U);     
     omega_f = graph_fourier_transform(omega, U);            
%     T_omega_f = fft(T_omega,[],3);
%     omega_f = fft(omega, [], 3);
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

iter=1;
while iter <=25
    fprintf('Sample--%f---Round--#%d\n', 2, iter);
    %[X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans * 1/k, Y_f_trans);
    [X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans, Y_f_trans);
    
    for i =1:k
        X_f(:,:,i) = X_f_trans(:,:,i)';
    end

    % Given X, do LS to get Y
    %[Y_f] = alter_min_LS_one_step(T_omega_f, omega_f * 1/k, X_f);
    [Y_f] = alter_min_LS_one_step(T_omega_f, omega_f, X_f);
    
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
T_est1= T_est;
T_est1 = T_est1*mean(T1(:))/mean(T_est1(:));  % into the same scale

temp = T1 - T_est1;   
error = norm(temp(:)) / norm(T1(:));
fprintf("error=%.8f\n", error);

for i=1:k
subplot(2,k,i);imagesc(abs(T1(:,:,i)));axis off;
colormap(gray);
subplot(2,k,i+k);imagesc(abs(T_est1(:,:,i)));axis off;
colormap(gray);
end
    

%profile viewer
