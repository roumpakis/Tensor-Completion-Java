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