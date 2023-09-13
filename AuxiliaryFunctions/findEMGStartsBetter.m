function [startValues, av, peaks, locs, thresh] = findEMGStartsBetter(emg, kinStarts, kinStops, thresh, startIdx, stopIdx)
%FINDEMGSTARTSBETTER Summary of this function goes here
%   Detailed explanation goes here
av = mean(emg);
% removed = zeros(height(emg),1);
% 
% for i = 1:length(emg)
%     [temp, locs] = rmoutliers(emg(:,i),"mean");
%     av(i) = mean(temp);
%     removed = removed + locs;
% end

startIdx = double(startIdx);
stopIdx = double(stopIdx);

peaks = zeros(1,length(kinStarts));
locs = zeros(1,length(kinStarts));

for i = 1:length(kinStarts)
    [pks, lcs] = findpeaks(av(kinStarts(i):kinStops(i)), 'SortStr','descend');
    peaks(i) = pks(1);
    locs(i) = kinStarts(i) + lcs(1) - startIdx;
end

maxVal = median(peaks);

thresh = thresh*maxVal;

window = 15;
starts = [];
% Finds the start index of the average EMG
for i = startIdx+window:stopIdx-window
    if median(av(i-window:i)) < thresh && median(av(i:i+window)) > thresh
        starts(end+1) = i;
    end
end

% Condense Multiples by middle-most value
boundaries = find(diff(starts)>5);
boundaries = [0, boundaries, length(starts)];

starts_new = [];
for i = 2:length(boundaries)
    starts_new(end+1) = min(starts(boundaries(i-1)+1:boundaries(i)));
end

% Filter start values based on distance from kinematic starting points
startVals = [];
valid = ones(1, length(starts_new));
halfdist = (kinStarts(cast(length(kinStarts), 'int32')/2+1)-kinStops(cast(length(kinStarts), 'int32')/2))/2;
for i = 1:length(starts_new)
    A = repmat(starts_new(i),[1 length(kinStarts)]);
    [minValue,closestIndex] = min(abs(A-kinStarts));
    if starts_new(i)>kinStops(closestIndex) || starts_new(i)<kinStarts(closestIndex)-halfdist
        valid(i) = 0;
    else
        startVals(end+1) = closestIndex;
    end
end

startVals = [starts_new(logical(valid)); startVals];
% disp(startVals);

% Finds the changepoints, where the average EMG starts increasing
changepts = [];
window = cast((kinStops(1)-kinStarts(1))/2, 'int64');
for i = 1:length(kinStarts)
    if i == 1
        s = findchangepts(av(startIdx:(kinStarts(i)+window)),'Statistic', 'mean');
    elseif i==length(kinStarts) && kinStarts(i)+window>stopIdx
        s = findchangepts(av((kinStarts(i)-window):stopIdx),'Statistic', 'mean')+kinStarts(i)-window;
    else
        s = findchangepts(av((kinStarts(i)-window):(kinStarts(i)+window)),'Statistic', 'mean')+kinStarts(i)-window;
    end
    changepts(end+1) = s;
end
% disp(changepts);

% Filter start values based on distance from changepoints and which
% kinematic it "belongs" to
usedKin = zeros(1, length(kinStarts));
filter = zeros(1, length(startVals));
starts = zeros(1, length(kinStarts));
for i = 1:length(startVals)
    kIndex = startVals(2, i);
    if usedKin(kIndex) == 0
        filter(i) = 1;
        starts(kIndex) = i;
        usedKin(kIndex) = 1;
    else
        if abs(changepts(kIndex)-startVals(1,i))<abs(changepts(kIndex)-startVals(1,starts(kIndex)))
            filter(starts(kIndex)) = 0;
            starts(kIndex) = i;
            filter(i) = 1;
        elseif abs(changepts(kIndex)-startVals(1,i))>abs(changepts(kIndex)-startVals(1,starts(kIndex)))
            filter(i) = 0;
        elseif abs(changepts(kIndex)-startVals(1,i))==abs(changepts(kIndex)-startVals(1,starts(kIndex)))
            if startVals(1,i)<startVals(1,starts(kIndex))
                filter(starts(kIndex)) = 0;
                starts(kIndex) = i;
                filter(i) = 1;
            else
                filter(i) = 0;
            end
        end
    end
end

startValues = startVals(1,logical(filter));

end

