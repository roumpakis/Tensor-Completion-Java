function y = SR(infile,outfile,windowSize,step,N,idxfile)
windowSize = str2num(windowSize);
step = str2num(step);
N = str2num(N);


 input = csvread(infile);
  idx = csvread(idxfile);
  input(idx) = NaN;
 dlmwrite('inNaN.csv', input, 'delimiter', ',', 'precision', 11); 
 [H, win_point_idx] = reshape_slidwin(input, windowSize, step);

 rec  = inpaintn(H);
 
 recVector =  reshape_slidwin_inv(rec, win_point_idx, N);
csvwrite(outfile,recVector);
y = recVector;

function [data_mtx, win_point_idx] = reshape_slidwin(raw_data, win_size, step_size)
% RESHAPE_SLIDWIN: Reshapes a vector into a matrix whose columns are the
%   elements of the sliding window.
%
%   INPUTS:
%       raw_data        - [vector] data to be reshaped
%       win_size        - [scalar] length of sliding window
%       step_size       - [scalar] step size to move the sliding window
%
%   OUTPUT:
%       data_mtx        - [2-D matrix] columns are the sliding windows
%       win_point_idx   - [2-D matrix] columns are the indices in original time-series 
%                                      contained in the sliding windows
%
% $ Date: 05/06/2013 10:10:51 $

% Input checking
if nargin < 3
    step_size = 1;
elseif nargin < 2
    error('At least two input arguments are required.');
end

numSamples  = numel(raw_data);
start_point = 1;
end_point   =  start_point + win_size - 1;
numWindows  = floor( (numSamples-start_point-win_size+1)/step_size + 1 );

raw_data = raw_data(:)';  % turn data into a row vector
win_point_idx = zeros(win_size, numWindows);
data_mtx = zeros(win_size, numWindows);
for t = 1:numWindows
    % Extract current window
    current_win = raw_data(start_point:end_point);
    %
    data_mtx(:,t) = current_win';
    % Slide window
    win_point_idx(:,t) = (start_point:end_point)';
    start_point = start_point + step_size;
    end_point   =  start_point + win_size - 1;
end
end
function y = inpaintn(x,n,y0,m)

% INPAINTN Inpaint over missing data in M-D array
%   Y = INPAINTN(X) replaces the missing data in X by extra/interpolating
%   the non-missing elements. The non finite values (NaN or Inf) in X are
%   considered as missing data. X can be any M-D array.
%   
%   NOTES:
%
%   INPAINTN uses an iterative process that converges toward the solution.
%   Y = INPAINTN(X,N) uses N iterations. By default, N = 100. If you
%   estimate that INPAINTN did not totally converge, increase N:
%   Y = INPAINTN(X,1000);
%
%   Y = INPAINTN(X,N,Y0) uses Y0 as initial guess. This could be useful if
%   you want to run the process a second time or if you have a good guess
%   of the final result. By default, INPAINTN makes a nearest neighbor
%   interpolation (by using BWDIST) to obtain a rough guess.
%


x = double(x);
if nargin==1 || isempty(n), n = 100; end

sizx = size(x);
d = ndims(x);
Lambda = zeros(sizx);
for i = 1:d
    siz0 = ones(1,d);
    siz0(i) = sizx(i);
     stelios=cos(pi*(reshape(1:sizx(i),siz0)-1)/sizx(i));
    
    Lambda = bsxfun(@plus,Lambda,cos(pi*(reshape(1:sizx(i),siz0)-1)/sizx(i)));
   
end
Lambda = 2*(d-Lambda);

% Initial condition
W = isfinite(x);
if nargin==3 && ~isempty(y0)
    y = y0;
    s0 = 3; % s = 10^s0
else
    if any(~W(:))
        [y,s0] = InitialGuess(x,isfinite(x));
    else
        y = x;
        return
    end
end
x(~W) = 0;

if isempty(n) || n<=0, n = 100; end

% Smoothness parameters: from high to negligible values
s = logspace(s0,-6,n); 

RF = 2; % relaxation factor

if nargin<4 || isempty(m), m = 2; end
Lambda = Lambda.^m;

 h = waitbar(0,'Inpainting...');
for i = 1:n
        Gamma = 1./(1+s(i)*Lambda);
        y = RF*idctn(Gamma.*dctn(W.*(x-y)+y)) + (1-RF)*y;
        waitbar(i/n,h)
end
close(h)

y(W) = x(W);

end

%% Initial Guess
function [z,s0] = InitialGuess(y,I)

if license('test','image_toolbox')
    %-- nearest neighbor interpolation
    [~,L] = bwdist(I);
    z = y;
    z(~I) = y(L(~I));
    s0 = 3; % note: s = 10^s0
else
    warning('MATLAB:inpaintn:InitialGuess',...
        ['BWDIST (Image Processing Toolbox) does not exist. ',...
        'The initial guess may not be optimal; additional',...
        ' iterations can thus be required to ensure complete',...
        ' convergence. Increase N value if necessary.'])
    z = y;
    z(~I) = mean(y(I));
    s0 = 6; % note: s = 10^s0
end

end


%% DCTN
function y = dctn(y)

%DCTN N-D discrete cosine transform.
%   Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
%   the same size as X and contains the discrete cosine transform
%   coefficients. This transform can be inverted using IDCTN.
%
%   Reference
%   ---------
%   Narasimha M. et al, On the computation of the discrete cosine
%   transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
%
%   Example
%   -------
%       RGB = imread('autumn.tif');
%       I = rgb2gray(RGB);
%       J = dctn(I);
%       imshow(log(abs(J)),[]), colormap(jet), colorbar
%
%   The commands below set values less than magnitude 10 in the DCT matrix
%   to zero, then reconstruct the image using the inverse DCT.
%
%       J(abs(J)<10) = 0;
%       K = idctn(J);
%       figure, imshow(I)
%       figure, imshow(K,[0 255])
%
%   -- Damien Garcia -- 2008/06, revised 2011/11
%   -- www.BiomeCardio.com --

y = double(y);
sizy = size(y);
y = squeeze(y);
dimy = ndims(y);

% Some modifications are required if Y is a vector
if isvector(y)
    dimy = 1;
    if size(y,1)==1, y = y.'; end
end

% Weighting vectors
w = cell(1,dimy);
for dim = 1:dimy
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);
end

% --- DCT algorithm ---
if ~isreal(y)
    y = complex(dctn(real(y)),dctn(imag(y)));
else
    for dim = 1:dimy
        siz = size(y);
        n = siz(1);
        y = y([1:2:n 2*floor(n/2):-2:2],:);
        y = reshape(y,n,[]);
        y = y*sqrt(2*n);
        y = ifft(y,[],1);
        y = bsxfun(@times,y,w{dim});
        y = real(y);
        y(1,:) = y(1,:)/sqrt(2);
        y = reshape(y,siz);
        y = shiftdim(y,1);
    end
end
        
y = reshape(y,sizy);

end

%% IDCTN
function y = idctn(y)

%IDCTN N-D inverse discrete cosine transform.
%   X = IDCTN(Y) inverts the N-D DCT transform, returning the original
%   array if Y was obtained using Y = DCTN(X).
%
%   Reference
%   ---------
%   Narasimha M. et al, On the computation of the discrete cosine
%   transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
%
%   Example
%   -------
%       RGB = imread('autumn.tif');
%       I = rgb2gray(RGB);
%       J = dctn(I);
%       imshow(log(abs(J)),[]), colormap(jet), colorbar
%
%   The commands below set values less than magnitude 10 in the DCT matrix
%   to zero, then reconstruct the image using the inverse DCT.
%
%       J(abs(J)<10) = 0;
%       K = idctn(J);
%       figure, imshow(I)
%       figure, imshow(K,[0 255])
%
%   See also DCTN, IDSTN, IDCT, IDCT2, IDCT3.
%
%   -- Damien Garcia -- 2009/04, revised 2011/11
%   -- www.BiomeCardio.com --

y = double(y);
sizy = size(y);
y = squeeze(y);
dimy = ndims(y);

% Some modifications are required if Y is a vector
if isvector(y)
    dimy = 1;
    if size(y,1)==1
        y = y.';
    end
end

% Weighing vectors
w = cell(1,dimy);
for dim = 1:dimy
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);
end

% --- IDCT algorithm ---
if ~isreal(y)
    y = complex(idctn(real(y)),idctn(imag(y)));
else
    for dim = 1:dimy
        siz = size(y);
        n = siz(1);
        y = reshape(y,n,[]);
        y = bsxfun(@times,y,w{dim});
        y(1,:) = y(1,:)/sqrt(2);
        y = ifft(y,[],1);
        y = real(y*sqrt(2*n));
        I = (1:n)*0.5+0.5;
        I(2:2:end) = n-I(1:2:end-1)+1;
        y = y(I,:);
        y = reshape(y,siz);
        y = shiftdim(y,1);            
    end
end
        
y = reshape(y,sizy);

end


function raw_data = reshape_slidwin_inv(data_mtx, win_point_idx, N)
% RESHAPE_SLIDWIN_INV: Reshapes a matrix whose columns are the elements of the sliding 
%   windows, to a vector by averaging the overlapping parts of the sliding windows.
%
%   INPUTS:
%       data_mtx        - [2-D matrix] columns are the sliding windows       
%       win_point_idx   - [2-D matrix] columns are the indices in original time-series 
%                                      contained in the sliding windows
%       N               - [scalar] number of samples in the original series
%
%   OUTPUT:
%       raw_data        - [vector] original data       
%
% $ Date: 05/06/2013 10:12:51 $

% Input checking
if nargin < 3
    error('Three input arguments are required.');
end

raw_data = zeros(1, N);
for k = 1:N
    % Find which windows contain the k-th point
    aux_idx_vec = find(win_point_idx == k);
    % Average overlapping parts
    raw_data(k) = mean(data_mtx(aux_idx_vec)); %#ok<*FNDSB>
end
end
end