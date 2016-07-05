%  A demo of Bayesian Robust CP Factorization for Incomplete Tensor Data
%  Author:  Qibin Zhao   2013

close all;
randn('state',1); rand('state',1); %#ok<RAND>

%% Generate a low-rank tensor
% Dimensions
DIM = [40,40,40];
R = 3; % ture rank for generating the tensor data
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
        part1 = [sin(temp); cos(temp); square(2*temp)]';
        part1 = zscore(part1);
        part2 = gaussSample(zeros(DIM(m),1), eye(DIM(m)), R-3)';
        Z{m} = [part1 part2];
        Z{m} = Z{m}(:,1:R);
    end 
end

% generate tensor from factor matrices
lambda = ones(1,R);
X = double(ktensor(lambda',Z));

%% Random missing values
ObsRatio = 0.5; % observation rate
Omega = randperm(prod(DIM)); 
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM); 
O(Omega) = 1;

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
Y = O.*Y;
SNR = 10*log10(var(O(:).*X(:)) / var(O(:).*(Y(:)-X(:))));

%% Run BRCPF
fprintf('------Bayesian Robust CP Factorization------------- \n');
tic
[model] = BayesRCP_TC(Y, 'obs', O, 'init', 'ml', 'maxRank', max([DIM,2*R]),  'maxiters', 100, 'verbose', 2);
t_total = toc;

% Performance evaluation
X_BRCP = double(ktensor(model.Z));
% S_hat = model.E;
err = X_BRCP(:) - X(:);
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));

% Report results
fprintf('\n------------------------BRCPF Result---------------------------------------------\n')
fprintf('Observation ratio = %g, SNR = %g, TrueRank=%d\n', ObsRatio, SNR, R);
fprintf('RRSE = %g, RMSE = %g, Estimated rank = %d, \nEstimated noise variance = %g, time = %g\n',...
    rrse, rmse, max(model.TrueRank), model.beta^(-1), t_total);
fprintf('-------------------------------------------------------------------------------\n')


%% Visualization of data and results
plotYXS(Y, X_BRCP, model.E, (Y-X_BRCP-model.E).*O);
factorCorr = plotFactor(Z,model.Z);

%%  CP-WOPT
% ncg_opts = ncg('defaults');
% ncg_opts.StopTol = 1.0e-6;
% ncg_opts.RelFuncTol = 1.0e-20;
% ncg_opts.MaxIters = 10^3;
% ncg_opts.DisplayIters = 50;
% [M,~,output] = cp_wopt(tensor(Y), tensor(O), R, 'init', 'nvecs', 'alg', 'ncg', 'alg_options', ncg_opts);
% X_hat = double(M);
% err = X_hat(:) - X(:);
% rmse = sqrt( mean(err.^2));
% rrse = sqrt(sum(err.^2)/sum(X(:).^2));
% fprintf('\n-------------CP-WOPT------------------------------------------\n')
% fprintf('RRSE = %g, RMSE = %g, \n', rrse, rmse);
% fprintf('-------------------------------------------------------------\n')
% 
% factorCorr = plotFactor(Z, M.U);


