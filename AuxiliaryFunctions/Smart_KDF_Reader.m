function [Features, Kinematics, NIPTime] = Smart_KDF_Reader(FBD_folder, varargin)

directories = string(ls(FBD_folder));
kdf_larry = contains(directories,"Train");
directories = directories(kdf_larry);
kdf_larry = contains(directories,".kdf");
directories = directories(kdf_larry);

if ~isempty(varargin)
    idx = cell2mat(varargin(1));
    if idx > length(directories) || idx <= 0
        error("File index out of range")
    end
end

if length(directories) >= 1
    if isempty(varargin)
        directories = directories(end);
    else
        directories = directories(idx);
    end
else
    error("No KDF files found");
end

kdfFile = convertStringsToChars(strtrim(strcat(FBD_folder, "\", directories)));

[Kinematics,Features,~,~,NIPTime] = readKDF(kdfFile);

Features = Features(193:end, :);  % Isolates the EMG Features (Not the USEA)
% EMGFeat_final = EMGFeat(SE,:);      % Isolates just the Single-Ended Features
   
%%trim the kinematics
for i = flip(1:height(Kinematics))
    if sum(Kinematics(i,:)) == 0
        Kinematics(i,:) = [];
    end
end

end