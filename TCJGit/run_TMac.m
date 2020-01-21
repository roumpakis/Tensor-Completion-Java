function T_hat = run_TMac(params)
  T = params.T;
  Idx = params.Idx;
  
  Omega = find(Idx);
  Ak = T(Omega);
  
  N = ndims(T);
  Nway = size(T); % dimension of tensor
  coreNway = round(0.15*[Nway(1),Nway(2),Nway(3)]);
  coreNway(coreNway==0) = 1;
  
%   G = tensor(randn(coreNway));
%   A = cell(1,ndims(G));
%   % randomly generate factor matrices
%   for i = 1:ndims(G)
%       A{i} = randn(Nway(i),coreNway(i));
%   end
%   % generate tensor
%   M = full(ttensor(G,A));
%   % M = tensor(M.data/max(M.data(:)));
%   N = ndims(M);
%   M = M.data;

  opts = [];
  opts.maxit = 200;
  opts.tol = 1e-5; % run to maxit by using negative tolerance  
  opts.alpha = [1/3, 1/3, 1/3];
%   opts.Mtr = M; % pass the true tensor to calculate the fitting
  
%   % rank_dec strategy   
%   opts.alpha_adj = 0;
%   opts.rank_adj = -1*ones(1,3);
%   opts.rank_min = 5*ones(1,3);
%   opts.rank_max = 20*ones(1,3);
%   EstCoreNway = round(1.25*coreNway);
 
  % rank_inc strategy
  opts.alpha_adj = 0;
  opts.rank_adj = 1*ones(1,3);
  opts.rank_inc = 1*ones(1,3);
  opts.rank_min = 1*ones(1,3); 
  opts.rank_max = round(1.5*coreNway);
  EstCoreNway = round(.75*coreNway); %ones(1,3);
  
  coNway = zeros(1,N);
  for n = 1:N
    coNway(n) = prod(Nway)/Nway(n);
  end
  
  % use random generated starting point
  %--- SRoub  addition
  minim = min(Ak(:));
  maxim = max(Ak(:));
  %---
  for i = 1:3
%       %--- SRoub  addition
      X0{i} = (maxim-minim)*randn(Nway(i),EstCoreNway(i)) + minim;
      Y0{i} = (maxim-minim)*randn(EstCoreNway(i),coNway(i)) + minim;
%       %---
%       X0{i} = randn(Nway(i),EstCoreNway(i));
%       Y0{i} = randn(EstCoreNway(i),coNway(i));
  end
  
  opts.X0 = X0; opts.Y0 = Y0;
%   [X_dec,Y_dec,Out_dec] = TMac(Ak,Omega,Nway,EstCoreNway,opts);
  [X_inc,Y_inc,Out_inc] = TMac(Ak,Omega,Nway,EstCoreNway,opts);
  
  % use the weighted sum of all mode matrix factorizations as estimated tensor
  T_hat = zeros(Nway);
  for i = 1:N
%     T_hat = T_hat+Out_dec.alpha(i)*Fold(X_dec{i}*Y_dec{i},Nway,i);
    T_hat = T_hat+Out_inc.alpha(i)*Fold(X_inc{i}*Y_inc{i},Nway,i);
  end
  
% Put original observations in their locations in the reconstructed tensor
T_hat(Omega) = T(Omega);

end