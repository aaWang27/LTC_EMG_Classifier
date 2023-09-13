function [data,labels,varargout] = formatData(Kinematics,Features,NIPTime,TrialStruct,DOFmap,mvmntsPerSet,varargin)
% Formats training and testing data for neural networks in Python.
%% Get Variable Inputs
try
    showPlot = varargin{1};
catch
    showPlot = 0;
end
if(showPlot)
    plotProg = 'training-progress';
else
    plotProg = 'none';
end
try
    checkpoints = varargin{2};
    cppath = '\\PNILABVIEW\PNILabview_R1\NeuralNetwork_DoNotDelete\Checkpoints\';
catch
    checkpoints = 0; %1 = save checkpoints
    cppath = '';
end
%% Select Data to Include
neuralFeatures = 0;  %1 = include neural,  0 = emg only
kinematicFeatures = 0;  %1 = include past kinematics, 0 = don't
PATIENCE = 10;
movementLabels = {'Thumb','Index','Middle','Ring','Little','ThumbInt','IndexInt','RingInt','LittleInt','WristFlex','WristDeviation','WristRotate'};
%% Align Data
badIdxs = [1:192] ; % ignore neural (ch 1:192) use emg only
alignMethod = 'trialByTrial'; % 'trialByTrial' or 'standard'
[Kinematics, Features] = alignTrainingData_aw(Kinematics, Features, badIdxs, mvmntsPerSet, alignMethod); %align data
%% Specify Parameters
numClasses = 4; %num DOFs
Features = Features(193:end,:);     %emg only
[numFeatures, ~] = size(Features);    %num features
KinLen = length(Kinematics);        %length of the total data recorded
%% VARIABLE NEURAL NETWORK PROPERTIES
% When splitting the training data into a testing and a training set, we
% have several variable options:
% First, we can choose to train on combo movements, or not.  Not training
% on combo movements allows us to see how well the algorithm generalizes.
% Second, we can use a certain percentage for training vs testing.
% Third, we choose what portion of the dataset we would like to designate
% as training.
trainCombo = 0;   %train on combos
testCombo = 0;  %test on combos
trainPercent = .75;  %75 percent
trainingType = 'shuffle';   % 'first' X percent. 'last' X percent. or 'shuffle' X percent.


% A sliding window is used to store previously recorded values. A larger
% window size uses more data points from the past to predict the future
windowSize = 25;
batchSize = 5;

% For the convolutional neural network, convolution is done on the input in
% the first convolutional layer.  The filter size determines how this
% convolution is done. The dimensionality is: [features, time]. A filter of
% size [1 5] will convolve a window of 5 previous time values and will not 
% convolve across any of the features.
filterSize = [3, 10];

% Given a windowsize of 10 and 720 features, the input will be of 
% size [720, 10].  With a filtersize of [1 5], the first convolution layer 
% will produce a feature map of size [720, 6].

% The number of filters specifies the number of feature maps that are
% produced. A single filter might develop to detect increases in EMG (where
% weights are laid out like [-1 -.5 0 .5 1]) or rapid changes over time (where
% weight are laid out like [1 -1 1 -1 1]).  If you include multiple
% filters, they will develop to explain more complexities of the data.
numFilters = 10;

% The initial learning rate for the CNN. If the learning rate is too low, 
% then training takes a long time. If the learning rate is too high, then 
% training might reach a suboptimal result.
learnRate = .001;

% An iteration is one step taken in the gradient descent algorithm towards
% minimizing the loss function using a mini-batch. An epoch is the full 
% pass of the training algorithm over the entire training set. More Epochs
% will result in longer training but potentially a more accurate network
maxEpoch = 2000;

inputSize = [numFeatures,windowSize,1]; %only one channel (i.e. not RGB image)

% Combine all the layers together in a |Layer| array.
%% Get Feature Sets for Training the Neural Network
% Convert Kinematics into Classes
[windows, labels] = windowMakerPython(Features, classificationArrayMaker(Kinematics), windowSize, batchSize);
labels = cell2mat(labels);
data = zeros(length(windows), height(Features), windowSize, 1);
for i = 1:length(windows)
    data(i,:,:,:) = windows{i};  
end
tempMap = zeros(1, 13);
for i = 1:length(DOFmap)
    tempMap(DOFmap(i)+1) = i;
end

%[data, labels] = balanceData(data,labels);
% cats = zeros(1, numClasses);
% s = size(data);
% for point = 1: s(1)
%     cats(1,tempMap(labels(point)+1)) = cats(1,tempMap(labels(point)+1)) +1;
% end
% total = sum(cats);
% for cl = 1:numClasses
%    fprintf("Percent that is class %d: %d \n", DOFmap(cl), (cats(1,cl)/total)*100);
% end

%% Variable outputs
varargout{1} = windowSize;
varargout{2} = numFeatures;