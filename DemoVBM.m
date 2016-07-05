
% A demo for video background modeling
%  Author:  Qibin Zhao   2014


%% load video data 
fpath = strcat('.', filesep, 'videos', filesep);
fname = 'hall';
% fname = 'WaterSurface';
% fname = 'ShoppingMall';
% fname = 'Lobby';

load(strcat(fpath,fname));


%%  BRCPF for video background and foreground modeling
Y = double(reshape(vid1, [imHeight*imWidth, 3, nFrames]));
Y = Y./255;
timecost =[];
IV =10;
if isequal(fname,'ShoppingMall') || isequal(fname,'Lobby')
    IV = 1;
end

tic
[model] = BayesRCP(Y, 'init', 'rand', 'maxRank', 10, 'maxiters', 20, 'initVar', IV, 'updateHyper', 'off',...
    'tol', 1e-3, 'dimRed', 1, 'verbose', 1);
timecost(1) = toc;
X = double(ktensor(model.Z)) * 255;
S = model.E * 255;


%% Visualization for results
X_BRCPF = reshape(X, [imHeight, imWidth, 3, nFrames]);
S_BRCPF = reshape(S, [imHeight, imWidth, 3, nFrames]);
showbgfg(vid1, X_BRCPF, S_BRCPF);









