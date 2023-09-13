% BEFORE YOU RUN THIS FILE, USE A DIFFERENT FILE TO TRAIN THE NN AND KALMAN
% FILTER
p = what();
p = p.path;
addpath(genpath("AuxiliaryFunctions"));
addpath(genpath("C:\Users\Aaron Wang\Box\JAGLAB\Projects\SmartHome\Controller\TestData"));

KDFFiles = {"TestData\20230314_SJ_Pilot\TrainingData_20230314-112821_113353.kdf"};
note = 'SJ_Pilot_03_14';

%% Options
savepath = 'Data\';
checkpoints = 0; %1 = save checkpoints
showPlot = 1;
Kinematics = [];     
Features = [];
Targets = [];
Kalman = [];
NIPTime = [];
CNNDOFmap = [0, 1, 2, 3];
for KDFFile = 1:length(KDFFiles)
    [Ki,F,T,Ka,NIP] = readKDF_jag(KDFFiles{KDFFile});
    Kinematics = [Kinematics, Ki];
    Features = [Features, F];
    Targets = [Targets, T];
    Kalman = [Kalman, Ka];
    NIPTime= [NIPTime, NIP];
    %KEFFile = regexprep(KDFFile,'.kdf','.kef');
    TrialStruct = 0;
end

%% Format Data
[data,labels,windowSize,numFeatures] = formatData(Kinematics,Features,NIPTime,TrialStruct,CNNDOFmap,20,showPlot,checkpoints);

%% Save File
% variable to save
windowSize;
numFeatures;
data;
labels;
% Save Data
directory = split(KDFFiles{1},'\');
filename = directory{end};
path = regexprep(filename, '.kdf', '');   %replaces .kdf with .mat in filename
dt = datestr(now,'dd-mmm-yyyy HH:MM:SS');
dt = regexprep(dt, ' ', '_'); 
dt = regexprep(dt, ':', '-'); 

% Save to .mat File
MATFile = [savepath path '_' note '.mat'];

save(MATFile,'data','labels','windowSize','numFeatures'); %saves relevant variables for online use

% Save to .csv File
% CSVDataFile = [savepath path '_' note '_Data_' dt '.csv'];
% CSVLabelsFile = [savepath path '_' note '_Labels_' dt '.csv'];
% 
% csvwrite(CSVDataFile,data);
% csvwrite(CSVLabelsFile,labels);