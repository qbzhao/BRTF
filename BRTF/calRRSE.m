

function rrse =  calRRSE(Xhat, X)

err = Xhat(:) - X(:);
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));