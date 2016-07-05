%% A demo of Bayesian Robust CP Factorization 
%  Author:  Qibin Zhao   2013

close all; 
randn('state',1); rand('state',1); %#ok<RAND>

%% Generate a low-rank tensor 
% Dimensions
DIM = [30,30,30];
R = 5; % ture rank for generating the tensor data
DataType = 2;

Z = cell(length(DIM),1);
if DataType ==1
    for m=1:length(DIM)
        Z{m} =  gaussSample(zeros(R,1), eye(R), DIM(m));
%         Z{m} =  gaussSample(zeros(DIM(m),1), eye(DIM(m)), R)';
    end
end
if DataType == 2      
    for m=1:length(DIM)   
        temp = linspace(0, m*2*pi, DIM(m));
        part1 = [sin(temp);  cos(temp); square(2*temp)]';
%         part1 = zscore(part1);        
        part2 = gaussSample(zeros(DIM(m),1), eye(DIM(m)), R-3)';        
        Z{m} = [part1, part2];        
        Z{m} = Z{m}(:,1:R);        
    end 
end

% generate tensor from factor matrices
lambda = ones(1,R);
X = double(ktensor(lambda',Z));


%% Generate dense noise 
SNR = 20;
sigma2 = var(X(:))*(1/(10^(SNR/10)));
GN =  sqrt(sigma2)*randn(DIM);

%% Generate sparse tensor
SparseRatio = 0.05;
Somega = randsample(prod(DIM), round(prod(DIM)*SparseRatio));
S = zeros(DIM);
% S(Somega) = 10*std(X(:))*(2*rand(length(Somega),1)-1); % sparse component
S(Somega) = max(X(:))*(2*rand(length(Somega),1)-1); % sparse component

%% Generate observation tensor Y
Y = X + S + GN;
SNR = 10*log10(var(X(:)) / var(Y(:)-X(:)));

%% Run BRCPF
fprintf('------Bayesian Robust Tensor Factorization---------- \n');
tic
[model] = BayesRCP(Y, 'init', 'ml', 'maxRank', max([DIM,2*R]), 'maxiters', 100,  'verbose', 2);
t_total = toc;

% Performance evaluation
X_hat = double(ktensor(model.Z));
% S_hat = model.E;
err = X_hat(:) - X(:);
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));

% Report results
fprintf('\n------------------------BRTF Result--------------------------------------------\n')
fprintf('SNR = %g, TrueRank=%d\n', SNR, R);
fprintf('RRSE = %g, RMSE = %g, Estimated rank = %d, \nEstimated noise variance = %g, time = %g\n',...
    rrse, rmse, max(model.TrueRank), model.beta^(-1), t_total);
fprintf('-------------------------------------------------------------------------------\n')

%% Visualization of data and results
plotYXS(Y, X_hat, model.E, (Y-X_hat-model.E));
factorCorr = plotFactor(Z,model.Z);

% 
%% CP-ALS
if 1    
    if R >= max(DIM)
        P = cp_als(tensor(Y),R,'init','random');
    else
        P = cp_als(tensor(Y),R,'init','nvecs');
    end
    X_hat = double(P);
    err = X_hat(:) - X(:);
    rmse = sqrt( mean(err.^2));
    rrse = sqrt(sum(err.^2)/sum(X(:).^2));
    % % Report results
    fprintf('\n-------------CP-ALS------------------------------------------\n')
    fprintf('RRSE = %g, RMSE = %g, \n', rrse, rmse);
    fprintf('-------------------------------------------------------------\n');    
% %     factorCorr = plotFactor(Z, P.U);    
end





