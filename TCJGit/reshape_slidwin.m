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

