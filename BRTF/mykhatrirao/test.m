

a = randn(40000,100);
b = randn(200,100);
c = randn(8000000,2);
A = {a,b};
C = reshape(c, [40000,200,2]);

%%
tic;
out1 = khatrirao(A,'r');
toc

%%
tic;
out2 = c'*khatrirao_fast(A,'r');
toc

%%   X * (kr(a,b))
tic
Anew = cell(1,100);
for i=1:100
    Anew{i} = cellfun(@(x) x(:,i), A,'UniformOutput',false);
end

X = tensor(C);
[out0 ]= cellfun(@(x) double(ttm(X, x, [1 2], 't')), Anew, 'UniformOutput',false);
out0= squeeze(cell2mat(out0))';
toc

isequal(out0,out2)



%%
out0 = khatrirao_space(A,C)';

%%
plot(out0(:)-out2(:))









%%
tic;
A = mat2cell(a,40000,ones(1,50));
B = mat2cell(b,200,ones(1,50));
% C = cellfun(@(x,y) KronProd({x',y'},[1,2])*randn(800,2), A,B);
C = cellfun(@(x,y) (KronProd({x',y'},[1,2])*randn(40000,200))', A,B, 'UniformOutput',false);
C = cell2mat(C);
toc
