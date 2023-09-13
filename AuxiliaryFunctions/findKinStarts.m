function [startValues, endValues, maxKinematicWindow, minDelayBetweenMovements] = findKinStarts(kin)
%FINDKINSTARTS Summary of this function goes here
%   Detailed explanation goes here
starts = [];
ends = [];
mask = diff(any(kin>0));


for i = 1:length(kin)-1
    if mask(i)==1
        starts(end+1) = i;
    elseif mask(i)==-1
        ends(end+1) = i+1;
    end
end

delays = [];
for i = 2:length(starts)
    delays(end+1) = starts(i) - ends(i-1);
end


startValues = starts;
endValues = ends;
maxKinematicWindow = max(abs(starts-ends));
minDelayBetweenMovements = min(delays);

end

