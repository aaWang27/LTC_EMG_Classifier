function badChans = findBadChannels(emg)
%FINDBADCHANNELS Summary of this function goes here
%   Detailed explanation goes here
removed = zeros(height(emg),1);
for i = 1:length(emg)
    [~, locs] = rmoutliers(emg(:,i),"mean");
    removed = removed + locs;
end

badChans = removed>length(emg)/4;
end

