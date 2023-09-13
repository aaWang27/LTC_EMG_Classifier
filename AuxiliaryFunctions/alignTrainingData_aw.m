function [XAligned, ZAligned, varargout] = alignTrainingData_aw(XOrig, ZOrig, badIdxs, method, numMvmnts, varargin)
% inputs
%   Xorig: 12 x n samples kinematic data movement cue
%   Zorig: 720 x n samples Feature data
%   BadChans: vector of bad chan idx
%   method: string 'standard' (correlation based) or 'trialByTrial'
% outputs:
% XAligned: 12 x n aligned kinematics
% ZAligned: 720 x n aligned features
% smw

% init
XAligned = XOrig;
ZAligned = ZOrig;

TrainZ(badIdxs, :) = 0;
switch method
    case 'standard'
        % find lag, apply to training data
        [Mvnts,Idxs,MaxLag,~,C] = autoSelectMvntsChsCorr_FD_jag(XOrig,ZOrig,0.4,badIdxs);
        ZAligned = circshift(ZOrig, MaxLag,2);
        varargout{1} = MaxLag;
        varargout{2} = C;
        
    case 'trialByTrial'
        [Mvnts,Idxs,MaxLag,~,C] = autoSelectMvntsChsCorr_FD_jag(XOrig,ZOrig,0.4,badIdxs);
        ZAligned = circshift(ZOrig, MaxLag,2);
        lags = zeros(1,60);
        
        % Find EMG and Kinematic Start values
        [kinStartValues, kinEndValues] = findKinStarts(XAligned);

        emgStartValues1 = findEMGStartsBetter(ZAligned, kinStartValues(1:numMvmnts), kinEndValues(1:numMvmnts), 0.2, 1, kinEndValues(numMvmnts)+1);
        emgStartValues2 = findEMGStartsBetter(ZAligned, kinStartValues(numMvmnts+1:2*numMvmnts), kinEndValues(numMvmnts+1:2*numMvmnts), 0.2, kinEndValues(numMvmnts)+2, kinEndValues(2*numMvmnts)+1);
        emgStartValues3 = findEMGStartsBetter(ZAligned, kinStartValues(2*numMvmnts+1:end), kinEndValues(2*numMvmnts+1:end), 0.2, kinEndValues(2*numMvmnts)+2, length(ZAligned));
        emgStartValues = [emgStartValues1 emgStartValues2 emgStartValues3];

        if length(kinStartValues) ~= length(emgStartValues)
            error("Number of detected EMG spikes differs from the number of expected kinematics");
        end
        
        % Align kinematics
        XAlignedShifted = XAligned;
        for i = 1:length(emgStartValues)
            lag = cast(emgStartValues(i)-kinStartValues(i), "int32");
            lags(i) = lag;
            if lag > 0
                for j = kinEndValues(i)+lag:-1:kinStartValues(i)+lag
                    XAlignedShifted(:, j) = XAligned(:, j-lag);
                end

                for j = kinStartValues(i)+lag-1:-1:kinStartValues(i)
                    XAlignedShifted(:, j) = 0;
                end
            else
                for j = kinStartValues(i)+lag:kinEndValues(i)+lag
                    XAlignedShifted(:, j) = XAligned(:, j-lag);
                end

                for j = kinEndValues(i)+lag+1:kinEndValues(i)
                    XAlignedShifted(:, j) = 0;
                end
            end
        end
        
        XAligned = XAlignedShifted;
        % note: could zap badKalmanIdxs at this point before sending to train

        varargout{1} = lags;
end