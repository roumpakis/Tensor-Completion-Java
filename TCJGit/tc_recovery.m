function recovery = tc_recovery(winSize, stepSize)



% This script demonstrates the use of tensor completion for the
% recovery of missing values in the received pressure data streams.
%
% The TMac algorithm is used for the tensor completion



%% Load pressure data
%load('data_2017-01-02_12');
%Pressure_data = P;  % each column correspond to a sensor
%sensor_ID = sensorID;

f = dir('*.csv'); % find all data files

 %add data to a matrix
T = [];
Mask = [];

for i =1:size(f)
   fname = f(i).name;
   data = csvread(fname);
   idx_values = find( isnan(data) ~= 1); %stream's real data position
   stream_mask = zeros(size(data)); %initialize mask with zeros
   stream_mask(idx_values) = 1;     % chabge positions with real data
   Mask(:,i) = stream_mask';         % add i-th stream;s mask on Mask table
   T(:,i) = data';                   %add data to i-th column
end


Data_Matrix = T;
idx_NaN = find( isnan(Data_Matrix) == 1); %missing values position
[N S] = size(Data_Matrix);
[nonZeroN,~] = size(idx_values);
fr = nonZeroN/(N*S);                    % calculate fill ratio



Data_Matrix_NaN = Data_Matrix; % matrix with NaN as missing values
Data_Matrix_Zero = Data_Matrix;  
Data_Matrix_Zero(idx_NaN)=0.0; % stream with zeros as missing values (TMac)

win_size = winSize;  % window size for Hankelization
step_size = stepSize;  % step size for Hankelization


%% Main
%err_ser_TMac = zeros(numel(fr),S);
% err_ser_tTNN = zeros(numel(fr),S);
%err_ser_interp = zeros(numel(fr),S);

    
    % Generate missing values
    mr = 1-fr;  % missing values ratio

    
    err_ser_TMac_MC = 0;



               %% Hankelization
        for s = 1:S
            cur_sensor = Data_Matrix_Zero(:,s);
            [data_Hankel, win_point_idx] = reshape_slidwin(cur_sensor, win_size, step_size);
            cur_sensor_mask = Mask(:,s);
            [mask_Hankel, mask_win_point_idx] = reshape_slidwin(cur_sensor_mask, win_size, step_size);
            Data_Hankel_Tensor(:,:,s) = data_Hankel;  % Hankelized data
            Data_winpoints_Hankel_Tensor(:,:,s) = win_point_idx;  % window points to be used for de-Hankelization
            Mask_Hankel_Tensor(:,:,s) = mask_Hankel;  % Hankelized mask
        end

   

        %% Tensor Completion
        params_tc.T = Data_Hankel_Tensor;
        params_tc.Idx = Mask_Hankel_Tensor;
        Data_Hankel_Tensor_TMac_hat = run_TMac(params_tc);
%         Data_Hankel_Tensor_tTNN_hat = run_tTNN(params_tc);

        %% Apply smoothing using inpaintn. Initialize using TC output
        for s = 1:S
            v = Data_Hankel_Tensor(:,:,s);
            NaN_idx = find(~Mask_Hankel_Tensor(:,:,s));
            v(NaN_idx) = NaN;
            y_TMac = inpaintn(v, 150, Data_Hankel_Tensor_TMac_hat(:,:,s));
            Data_Hankel_Tensor_TMac_hat(:,:,s) = y_TMac;

        end

        %% De-Hankelization
        Data_Matrix_TMac_rec = zeros(N,S);
%         Data_Matrix_tTNN_rec = zeros(N,S);
        for s = 1:S
            cur_sensor_TMac_hat = reshape_slidwin_inv(Data_Hankel_Tensor_TMac_hat(:,:,s), Data_winpoints_Hankel_Tensor(:,:,s), N);
            Data_Matrix_TMac_rec(:,s) = cur_sensor_TMac_hat(:);

        end
        
       
        
        %% Write each reconstructed stream to file       
        for s = 1:S
            rec_stream =  Data_Matrix_TMac_rec(:,s);
            rec_name = strcat('rec_',f(s).name);
          csvwrite(rec_name,rec_stream);  

        end


recovery = Data_Matrix_TMac_rec;

end