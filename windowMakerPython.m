function [windows, out_labels] = windowMakerPython(EMG_data, label_array, window_size, batch_size)
%Inputs: Raw Kinematic (emg) data, a label mask array generated usually by
%classificationArraymaker, and a window size to determine the size of each
%window
%Outputs: Sliced and windowified EMG data, labeled array of output 
len = width(EMG_data);

windows = cell(1, len);
out_labels = cell(1, len);

for i = (window_size):len
    windows{i} = EMG_data(:, (i-window_size+1):i);
    if length(label_array) ~= 1
        out_labels{i} = label_array(i);
    end
end

windows = windows(window_size:end);
if length(label_array) ~= 1
    out_labels = out_labels(window_size:end);
else
    out_labels = 0;
end

% windows = cell(1, length(temp_windows));
% out_labels = zeros(batch_size, length(temp_windows));
% 
% for i = (batch_size):length(temp_windows)
%     windows{i} = temp_windows((i-batch_size+1):i);
%     if length(label_array) ~= 1
%         out_labels(:,i) = cell2mat(labels(i-batch_size+1:i));
%     end
% end
% 
% windows = windows(batch_size:end);
% if length(labels) ~= 1
%     out_labels = out_labels(:, batch_size:end);
% else
%     out_labels = 0;
% end
end 