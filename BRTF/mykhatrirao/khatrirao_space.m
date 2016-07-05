function P = khatrirao_space(A,X)


%   kr(A{N},...,A{1})' * X_(N)'

N = length(A);
% xdim = ndims(X);

[~,col] = size(A{1});

Anew = cell(1,col);
for i=1:col
    Anew{i} = cellfun(@(x) x(:,i), A,'UniformOutput',false);
end

X = tensor(X);

P = cellfun(@(x) double(ttm(X, x, [1:N], 't')), Anew, 'UniformOutput',false);
P= squeeze(cell2mat(P));





