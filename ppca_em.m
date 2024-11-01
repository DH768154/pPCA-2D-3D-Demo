function [W, mu, para] = ppca_em(X, k, tol, max_iter,W0,s0)
%%
% X: d*n Matrix
% k: scaler, redused dimention
% W: initial W, d*k Matrix
% s: scaler

W = W0;
s = s0;

%%
dotsum = @(A,B) A(:)'*B(:);
% sum(A.*B,'all'), numpy use this way is faster
% trace(A*B')
% sum(||X_n||^2) = sum(X.^2,'all') = trace(X*X') = X(:)'*X(:)

[D, n] = size(X);
mu = mean(X, 2);
X = X - mu;

% cov of X, constant, not nesseary calculate in loop
covX = X*X';

% trace of X*X', this is sum(||X_n||^2)
traceS = trace(covX);

%% Loop

% W'*W will updated in loop, this is initial value
Q = W'*W;

iter = max_iter;
loglk = NaN(max_iter+1,1);

for i = 1:max_iter

    % M is used to calculate inv(M)
    % M = Q + s*eye(k);
    U = chol(Q + s*eye(k));

    %% Estimate log likelihood

    C = W*W' + s*eye(D);
    H = W/U;
    covH = H*H';
    tr_SiC = 1/s/n* (traceS-dotsum(covX,covH));

    % log(det(C)), use svd for det=0
    detC = det(C);
    if detC==0
        [~,ss,~] = svd(C,'econ');
        ss = diag(ss);
        logdetC = sum(log(ss(ss~=0))); % deal with det(C)==0
    else
        logdetC = log(detC); % dirctly calculate log(det(C))
    end

    loglk(i) = -(D*log(2*pi) + logdetC + tr_SiC)*n/2;
    
    %% Break
    % 5 points not change, break 
    if i>=5 && mean(abs(diff(loglk(i-4:i))))<=tol
        iter = i;
        loglk = loglk(1:iter);
        break
    end

    %% E step & M step

     % Use chol solve inv(M)
    iM = U \ (U'\eye(size(U)));

    % 如果噪声是0  ，Ez = inv(W'*W    )*W' * X = pinv(W) * X, 也即是X在低纬度下的投影。
    % 如果噪声不是0，Ez = inv(W'*W+s*I)*W' * X, Ez代表给定噪声模型情况下的后验期望，
    % 它考虑了噪声的影响，体现了观测数据 X 在低维空间中的一个最优的估计
    Ez = iM * W'*X;

    % 同样道理，如果噪声是0，Ezzt = Ez*Ez' 就是X在低纬度投影的协方差。
    % 考虑了噪声的影响，Ezzt是 X 在低维空间中的最优的估计的协方差
    Ezzt = s * iM * n + Ez * Ez';

    % W: 投影矩阵
    W = X*Ez'/ Ezzt;
    Q = W'*W; % 计算Q是因为这个值在下一个循环还要用到

    % s: 噪声的协方差
    s = (traceS - 2*dotsum(W*Ez,X) + dotsum(Ezzt,Q)) /n/D;

end

if i==max_iter
    warning('Reach Max Iteration %.0f',max_iter)
end

%% Output Parameter

para.Ez = Ez; % 考虑噪声情况下，高维度数据X投影到低纬度的值
para.Covz = Ezzt; % 考虑噪声情况下，Ez的协方差。理想情况下是eye
para.iter = iter; % iteration
para.s = s; % cov for noise
para.loglk = loglk(end); % likelihood
para.loglk_all = loglk; % likelihood

end

