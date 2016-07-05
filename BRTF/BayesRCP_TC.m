function [model] = BayesRCP_TC(Y, varargin)
%  [model] = BayesRCP_TC(Y, 'PARAM1', val1, 'PARAM2', val2, ...)
%
%  INPUTS
%     Y              - input tensor
%     'obs'          - binary (0-1) indicator tensor of same size as Y 
%                      (0: missing, 1: observed)
%                      (default:  ones(size(Y)) )
%     'init'         - Initialization method.
%                     - 'ml'  : Apply SVD to Y and initialize factor matrices (default)
%                     - 'rand': initialize factor matrices with random matrices
%     'maxRank'      - The initial CP rank 
%                    - max(size(Y))  (default)
%     'dimRed'       - 1: Remove zero components automaticly (default)
%                    - 0: Keep number of components as the initialized value 
%     'initVar'      - Initialization of variance of outliers (default: 1)
%     'updateHyper'  - Optimization of top level parameter 
%                    - 'on' (default)
%                    - 'off' 
%     'maxiters'     - maximum number of iterations (default: 100)
%     'tol'          - lower band change tolerance for convergence dection     
%                      (default: 1e-6)
%     'predVar'      - Predictive confidence   
%                         - 1:  compute and output
%                         - 2:  fast computation
%                         - 0:  does not compute (default)
%     'verbose'      - visualization of results.
%                       - 0: no any 
%                       - 1: text (default)
%                       - 2: image plot 
%                       - 3: hinton plot
%   OUTPUTS
%      model         - Model parameters and hyperparameters
%
%   Example:
%
%        [model] = BayesRCP_TC(Y, 'obs', O, 'init', 'ml', 'verbose', 2);
%
%
% < Bayesian Robust CP Factorization for Tensor Completion >
% Copyright (C) 2013  Qibin Zhao
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters from input or by using defaults
dimY = size(Y);
N = ndims(Y);

ip = inputParser;
ip.addParamValue('obs', ones(dimY), @(x) (isnumeric(x) || islogical(x)) );
ip.addParamValue('init', 'ml', @(x) (iscell(x) || ismember(x,{'ml','rand'})));
ip.addParamValue('maxRank', max(dimY), @isscalar);
ip.addParamValue('initVar', 1, @isscalar);
ip.addParamValue('dimRed', 1, @isscalar);
ip.addParamValue('maxiters', 100, @isscalar);
ip.addParamValue('tol', 1e-6, @isscalar);
ip.addParamValue('predVar', 0, @isscalar);
ip.addParamValue('X_true', [], @isnumeric);
ip.addParamValue('verbose', 1, @isscalar);
ip.addParamValue('updateHyper', 'on', @ischar);
ip.parse(varargin{:});

O     = ip.Results.obs;
init  = ip.Results.init;
R   = ip.Results.maxRank;
maxiters  = ip.Results.maxiters;
tol   = ip.Results.tol;
X_true   = ip.Results.X_true;
verbose  = ip.Results.verbose;
DIMRED   = ip.Results.dimRed;
initVar   = ip.Results.initVar;
updateHyper = ip.Results.updateHyper;
predVar = ip.Results.predVar;


%% Initialization
randn('state',1); rand('state',1); %#ok<RAND>

Y = tensor(Y.*O);
O = tensor(O);
nObs = sum(O(:));
LB = 0;

a_gamma0     = 1e-6;
b_gamma0     = 1e-6;
a_beta0      = 1e-6;
b_beta0      = 1e-6;
a_alpha0     = 1e-6;
b_alpha0     = 1e-6;

gammas = (a_gamma0+eps)/(b_gamma0+eps)*ones(R,1);
beta = (a_beta0+eps)/(b_beta0+eps);
alphas = initVar.^(-1)*ones(dimY).*((a_alpha0+eps)/(b_alpha0+eps));

E = alphas.^(-0.5).*randn(dimY);
Sigma_E = alphas.^(-1).*ones(dimY);


if iscell(init)
    Z = init;
    if numel(Z) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = 1:N;
        if ~isequal(size(Z{n}),[size(Y,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    switch init,
        case 'ml'    % Maximum likelihood
            Z = cell(N,1);
            ZSigma = cell(N,1);
            if ~isempty(find(O==0))
                Y(find(O==0)) = sum(Y(:))/nObs;
            end
            for n = 1:N
                ZSigma{n} = (repmat(diag(gammas.^(-1)), [1 1 dimY(n)]));
                [U, S, V] = svd(double(tenmat(Y,n)), 'econ');
                if R <= size(U,2)
                    Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
                else
                    Z{n} = [U*(S.^(0.5)) randn(dimY(n), R-size(U,2)).*((a_gamma0/b_gamma0).^(-0.5))];
                end
            end
            Y = Y.*O;
        case 'rand'   % Random initialization
            Z = cell(N,1);
            ZSigma = cell(N,1);
            dvar = sqrt(sum(Y(:).^2)/nObs);
            for n = 1:N
%                   Z{n} = randn(dimY(n),R)*diag(gammas.^(-0.5));
                Z{n} = randn(dimY(n),R)*dvar;
                ZSigma{n} = (repmat(diag(gammas.^(-1)), [1 1 dimY(n)]));                
                %   ZSigma{n} = (repmat(dvar*eye(R,R), [1 1 dimY(n)]));
            end
    end
end

%% Create figures
if verbose >1,
    scrsz = get(0,'ScreenSize');
    h1 = figure('OuterPosition',[scrsz(3)*0.2 scrsz(4)*0.5 scrsz(3)*0.6 scrsz(4)*0.4]);   
    set(0,'CurrentFigure',h1);
    switch verbose,
        case 2,
            subplot(2,3,1); imagesc(Z{1}); title('Factor-1'); ylabel('Data dimensions');
            subplot(2,3,2); imagesc(Z{2}); title('Factor-2'); xlabel('Latent dimensions');
            if N>=3, subplot(2,3,3); imagesc(Z{3}); title('Factor-3');end
        case 3,
            subplot(2,3,1); hintonDiagram(Z{1}); title('Factor-1'); ylabel('Data dimensions');
            subplot(2,3,2); hintonDiagram(Z{2}); title('Factor-2'); xlabel('Latent dimensions');
            if N>=3, subplot(2,3,3); hintonDiagram(Z{3}); title('Factor-3'); end
    end
    subplot(2,3,4); bar(gammas); title('Posterior mean of \lambda'); xlabel('Latent components'); ylabel(''); axis tight;
    subplot(2,3,5); plot(LB, '-r.','LineWidth',1.5,'MarkerSize',10 ); title('Lower bound'); xlabel('Iteration');  grid on;
    subplot(2,3,6); plotGamma(a_beta0, a_beta0); title('Posterior pdf'); xlabel('Noise precision \tau');grid on;
    h2 = figure('OuterPosition',[scrsz(3)*0.2 scrsz(4)*0.3  scrsz(3)*0.6 scrsz(4)*0.2]);
    imagesc(reshape(E,[dimY(1), prod(dimY(2:end))])); title('Sparse outliers');
    %     X = double(ktensor(Z));
    %     subplot(1,2,1); voxel3(abs(X),'thresh',0, 'degree',5);  title('Low Rank Tensor','Fontsize',12); xlabel(''); ylabel(''); zlabel('');set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);set(gca,'zticklabel',[]);
    %     subplot(1,2,2); voxel3(abs(E),'thresh',0.1, 'degree',5);  title('Sparse Outliers','Fontsize',12); xlabel(''); ylabel(''); zlabel(''); set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);set(gca,'zticklabel',[]);
    drawnow;
end


%% Model learning

% --------- E(aa') = cov(a,a) + E(a)E(a')----------------
EZZT = cell(N,1);
for n=1:N
    EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))' + khatrirao_fast(Z{n}',Z{n}')';
end

for it=1:maxiters,
    %% Update factor matrices
    Aw = diag(gammas);
    for n=1:N
        % compute E(Z_{\n}^{T} Z_{\n})
        %         Eslash = khatrirao(EZZT{[1:n-1, n+1:N]},'r');
        ENZZT = reshape(khatrirao_fast(EZZT{[1:n-1, n+1:N]},'r')' * double(tenmat(O,n)'), [R,R,dimY(n)]);
        % compute E(Z_{\n})
        %         Fslash = khatrirao(Z{[1:n-1, n+1:N]},'r');
        FslashY = khatrirao_fast(Z{[1:n-1, n+1:N]},'r')' * tenmat((Y-E).*O, n)';
        for i=1:dimY(n)
            ZSigma{n}(:,:,i) = (beta * ENZZT(:,:,i) + Aw)^(-1);
            Z{n}(i,:) = (beta * ZSigma{n}(:,:,i) * FslashY(:,i))';
        end
        EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]) + khatrirao_fast(Z{n}',Z{n}'))';
    end
    
    %% Update latent tensor X
    X = double(ktensor(Z));
    
    %% Update hyperparameters gamma
    a_gammaN = (0.5*sum(dimY) + a_gamma0)*ones(R,1);
    b_gammaN = 0;
    for n=1:N
        b_gammaN = b_gammaN + diag(Z{n}'*Z{n}) + diag(sum(ZSigma{n},3));
    end
    b_gammaN = b_gamma0 + 0.5.* b_gammaN;
    gammas = a_gammaN./b_gammaN;
    
    %% update noise beta
    %  The most time and space consuming part
    if 0 % save time but large space needed
        EX2 =  O(:)' * khatrirao_fast(EZZT,'r') * ones(R*R,1);
    else  % save space but slow
        temp1 = cell(N,1);
        EX2 =0;
        for i =1:R
            for n=1:N
                temp1{n} = EZZT{n}(:,(i-1)*R+1: i*R);
            end
            EX2 = EX2 + O(:)' * khatrirao(temp1,'r')* ones(R,1);
        end
    end
    EE2 = sum((E(:).^2 + Sigma_E(:)).*O(:));
    err = Y(:)'*Y(:) - 2*Y(:)'*X(:) -2*Y(:)'*E(:) + 2*X(:)'*E(:) + EX2 + EE2;
    a_betaN = a_beta0 + 0.5*nObs;
    b_betaN = b_beta0 + 0.5*err;
    beta = a_betaN/b_betaN;
    
    %% Update sparse matrix E
    Sigma_E = double(O)./(alphas+beta);
    E = double(beta*Sigma_E.*(Y-X).*O);
    
    %% Update the alphas
    inf_flag = 1;
    if inf_flag == 1,
        % Standard
        a_alphaN = a_alpha0 + 0.5*(double(O));
        b_alphaN = b_alpha0 + 0.5*(E.^2 + Sigma_E);
        alphas = a_alphaN./b_alphaN;
    elseif inf_flag == 2,
        % Mackay
        a_alphaN = a_alpha0 + 1-alphas.*Sigma_E;
        b_alphaN = b_alpha0 + E.^2 + eps;
        alphas = a_alphaN./b_alphaN;
    end
    
    %% Lower bound
    temp1 = -0.5*nObs*safelog(2*pi) + 0.5*nObs*(psi(a_betaN)-safelog(b_betaN)) - 0.5*(a_betaN/b_betaN)*err;
    temp2 =0;
    for n=1:N
        temp2 = temp2 + -0.5*R*dimY(n)*safelog(2*pi) + 0.5*dimY(n)*sum(psi(a_gammaN)-safelog(b_gammaN)) -0.5*trace(diag(gammas)*sum(ZSigma{n},3)) -0.5*trace(diag(gammas)*Z{n}'*Z{n});
    end
    temp3 = sum(-safelog(gamma(a_gamma0)) + a_gamma0*safelog(b_gamma0) -  b_gamma0.*(a_gammaN./b_gammaN) + (a_gamma0-1).*(psi(a_gammaN)-safelog(b_gammaN)));
    temp4 = -safelog(gamma(a_beta0)) + a_beta0*safelog(b_beta0) + (a_beta0-1)*(psi(a_betaN)-safelog(b_betaN)) - b_beta0*(a_betaN/b_betaN);
    temp5 = 0.5*R*sum(dimY)*(1+safelog(2*pi));
    for n=1:N
        for i=1:size(ZSigma{n},3)
            temp5 = temp5 + 0.5*safelog(det(ZSigma{n}(:,:,i))) ;
        end
    end
    temp6 = sum(safelog(gamma(a_gammaN)) - (a_gammaN-1).*psi(a_gammaN) -safelog(b_gammaN) + a_gammaN);
    temp7 = safelog(gamma(a_betaN)) - (a_betaN-1)*psi(a_betaN) -safelog(b_betaN) + a_betaN;
    
    temp = psi(a_alphaN) - safelog(b_alphaN);
    temp8 = -0.5*nObs*safelog(2*pi) + 0.5*(temp(:)'*O(:)) - 0.5*(E(:).^2 + Sigma_E(:))'*alphas(:);
    temp9 = -nObs*safelog(gamma(a_alpha0)) + nObs*a_alpha0*safelog(b_alpha0) + ((a_alpha0-1)*(temp(:)) - b_alpha0*alphas(:))'*O(:);
    
    temp10 = 0.5*sum(safelog(Sigma_E(:))) + 0.5*nObs*(1+safelog(2*pi));
    temp11 = safelog(gamma(a_alphaN)) - (a_alphaN-1).*psi(a_alphaN) -safelog(b_alphaN) + a_alphaN;
    temp11 = temp11(:)'*O(:);
    
    LB(it) = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8 + temp9 + temp10 + temp11;
    
    %% update top level hyperparameters
    if strcmp( updateHyper, 'on')
        if it>5
            aMean = alphas(:)'*O(:)/nObs;
            bMean = (psi(a_alphaN(:)) - safelog(b_alphaN(:)))'*O(:)/nObs;
            ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
            a_alpha0 = fminsearch(ngLB,a_alpha0);
            %       a_alpha0 = 0.55/(log(aMean)-bMean);
            b_alpha0 = a_alpha0/aMean;
            % % % % % % %       ngLB = @(x) log(gamma(x(1))) - x(1).*log(x(2)) - (x(1)-1)*bMean + x(2)*aMean;
            
            
            %     % update the top level noise parameter
            %       aMean = beta;
            %       bMean = psi(a_betaN) - safelog(a_betaN);
            %       ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
            %       a_beta0 = fminsearch(ngLB,a_beta0);
            %       b_beta0 = a_beta0/aMean;
            
            %     % update the top level gamma parameter
            %       aMean = mean(gammas);
            %       bMean = mean(psi(a_gammaN) - safelog(b_gammaN));
            %       ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
            %       a_gamma0 = fminsearch(ngLB,a_gamma0);
            %       b_gamma0 = a_gamma0/aMean;
        end
    end
    
    %% Prune irrelevant dimensions?
    Zall = cell2mat(Z);
    comPower = diag(Zall' * Zall);
    comTol = sum(dimY)*eps(norm(Zall,'fro'));
    rankest = sum(comPower> comTol );
    if max(rankest)==0
        error('Rank becomes 0');  
    end
    if DIMRED==1  && it >=2,
        if R~= max(rankest)
            indices = comPower > comTol;
            gammas = gammas(indices);
            temp = ones(R,R);
            temp(indices,indices) = 0;
            temp = temp(:);
            for n=1:N
                Z{n} = Z{n}(:,indices);
                ZSigma{n} = ZSigma{n}(indices,indices,:);
                EZZT{n} = EZZT{n}(:, temp == 0);
            end
            R = rankest;            
        end
    end
     
    
    %% visualize online results
    if verbose >1 ,
        set(0,'CurrentFigure',h1);
        switch verbose,
            case 2,
                set(0,'CurrentFigure',h1);
                subplot(2,3,1); imagesc(Z{1}); title('Factor-1'); ylabel('Data dimensions');
                subplot(2,3,2); imagesc(Z{2}); title('Factor-2'); xlabel('Latent dimensions');
                if N>=3, subplot(2,3,3); imagesc(Z{3}); title('Factor-3'); end
            case 3,
                set(0,'CurrentFigure',h1);
                subplot(2,3,1); hintonDiagram(Z{1}); title('Factor-1'); ylabel('Data dimensions');
                subplot(2,3,2); hintonDiagram(Z{2}); title('Factor-2'); xlabel('Latent dimensions'); box on;
                if N>=3, subplot(2,3,3); hintonDiagram(Z{3}); title('Factor-3'); end
            case 4,
                set(0,'CurrentFigure',h1);
                temp = repmat(1:R,[dimY(1),1]);
                subplot(2,3,1); plot(Z{1}+temp); title('Factor-1'); ylabel('Data dimensions'); axis tight;
                temp = repmat([1:R].*max(std(Z{2}))*2,[dimY(2),1]);
                subplot(2,3,2); plot(Z{2}+temp); title('Factor-2'); xlabel('Latent dimensions');axis tight;
                if N>=3, temp = repmat([1:R].*max(std(Z{3}))*2,[dimY(3),1]); subplot(2,3,3); plot(Z{3}+temp); title('Factor-3');axis tight; end
        end
        subplot(2,3,4); bar(gammas); title('Posterior mean of \lambda'); xlabel('Latent components'); ylabel(''); axis tight;
        subplot(2,3,5); plot(LB, '-r.','LineWidth',1.5,'MarkerSize',10 ); title('Lower bound'); xlabel('Iteration');  grid on;
        subplot(2,3,6); plotGamma(a_betaN, b_betaN); title('Posterior pdf'); xlabel('Noise precision \tau');grid on;
        
        set(0,'CurrentFigure',h2);
        imagesc(reshape(E,[dimY(1), prod(dimY(2:end))])); title('Sparse outliers');
%       plot(E(:));
%       subplot(1,2,1); voxel3(abs(X),'thresh',0, 'degree',5);  title('Low Rank Tensor','Fontsize',12); xlabel(''); ylabel(''); zlabel('');set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);set(gca,'zticklabel',[]);
%       subplot(1,2,2); voxel3(abs(E),'thresh',0, 'degree',5);  title('Sparse Outliers','Fontsize',12); xlabel(''); ylabel(''); zlabel(''); set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);set(gca,'zticklabel',[]);
        drawnow;
    end
    
    %% Display progress
    if it>2
        converge = -1*((LB(it) - LB(it-1))/LB(2));
    else
        converge =inf;
    end
    if verbose,
        if ~isempty(X_true)
            err = X(:) - X_true(:);
            err = err.^2;
            fprintf('it %d: mse = %g, LB = %g, conv = %g, R = %d \n', it, mean(err), LB(it), converge, rankest);
        else
            fprintf('it %d: LB = %g, conv = %g, mse = %g, R = %d \n', it, LB(it), converge, err/nObs, rankest);
        end
    end
    %% Convergence check
    if it>5 && abs(converge) < tol
        disp('\\\======= Converged===========\\\');
        break;
    end
end

%% Predictive distribution
switch predVar
    case 1
        Xvar =  tenzeros(size(Y));
        for n=1:N
            Xvar = tenmat(Xvar,n);
            Fslash = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
            if 1
                temp1 = double(tenmat(tensor(ZSigma{n}),3));
                temp2 = khatrirao_fast(Fslash', Fslash');
                Xvar(:,:) = Xvar(:,:) + temp1*temp2;
            else
                % ---  slow computation ------
                for i=1:size(Xvar,1)     %#ok
                    Xvar(i,:) = Xvar(i,:) + diag(Fslash * ZSigma{n}(:,:,i) *Fslash')';
                end
                % ---  slow computation ------
            end
            Xvar = tensor(Xvar);
        end
        Xvar = Xvar + beta^(-1);
        Xvar = Xvar.*(2*a_betaN)/(2*a_betaN-2);
    case 2
        % Better for saving memory 
        Xvar = double(ktensor(EZZT))- X.^2;
        Xvar = Xvar + beta^(-1);
    otherwise
        Xvar =[];
end

%% Output
model.Z = Z;
model.ZSigma = ZSigma;
model.gammas = gammas;
model.E = E;
model.Sigma_E = Sigma_E;
model.beta = beta;
model.Xvar = double(Xvar);
model.TrueRank = rankest;
model.LowBound = max(LB);


function y = safelog(x)
x(x<1e-300)=1e-200;
x(x>1e300)=1e300;
y=log(x);

