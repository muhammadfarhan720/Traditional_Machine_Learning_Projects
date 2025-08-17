%% MIT_ref_maker.m
% Jeremy Decker
% 11/5/2022
% This script will take in the subject summary sheets from the MIT dataset,
% and create a reference sheet containing the following:
% 1. File of Origin(Full path from a project folder into the MIT database)
% 3. Seizure Start (Relative Time)
% 4. Seizure Duration 
% 6. Seizure Number in File
% 5. File Summary, for later(I guess?)
% This file will be passed into the EDF_Slicer File for use later

%% Choose a reference file
already_ref_file = exist('ref_file', 'var');
if already_ref_file == 0
    [fpath, foldpath] = uigetfile("", "Please select a file for summary file analysis");
    ref_path = strcat(foldpath,fpath);
    ref_file = readtable(ref_path);
end
% initialize a total seizures count
total_seizures = 1;
%eeglab;
%% Initialize a loop that opens each text file for analysis. 
for i = 1:length(ref_file.sum_file)
    file_seizures = 1;
    curr_file = ref_file.sum_file(i);
    curr_sum = fopen(string(curr_file)); % Already fully pathed. 
    [curr_path, ~, ~] = fileparts(curr_file);
    % Initialize the read_while loop for the file 
    tline1 = fgetl(curr_sum);
    disp(tline1) % Sanity Check
    while tline1 ~= -1 % Look for the end of file line, basically
        %disp(tline1)
        if size(tline1, 2) >= 11 && strcmp("File Name", tline1(1,1:9))
            %disp("Found New Filename")
            curr_filename = tline1(1, 12:length(tline1));
            disp(curr_filename)
        end
        if size(tline1, 2) >= 20 & regexp(tline1, "Start Time") & regexp(tline1, "Seizure")
            % Seizure found for the file, add the start and end times
            mes = sprintf("Seizure Found! Total Seizures %i, File Seizures: %i", total_seizures, file_seizures);
            disp(mes)
            % Use a regular expression to pull the numbers out of the line
            start_time = str2double(regexp(tline1, '\d*', 'match'));
            start_time = start_time(length(start_time));
            % Get the next line of the file
            tline1 = fgetl(curr_sum);
            stop_time = str2double(regexp(tline1, '\d*', 'match'));
            stop_time = stop_time(length(stop_time));
            % Conver them to integers/doubles?
            duration = stop_time-start_time;
            disp("Adding storm to structure")
            storms_sheet(total_seizures).eeg = strcat(curr_path, '/', curr_filename);
            storms_sheet(total_seizures).start = start_time;
            storms_sheet(total_seizures).duration = duration;
            storms_sheet(total_seizures).sum_file_seizures = file_seizures;
            storms_sheet(total_seizures).sum_file = curr_file;
            total_seizures = total_seizures+1;
            file_seizures = file_seizures+1;
        end
        % Get the next line in the file
        tline1 = fgetl(curr_sum);
        if size(tline1, 1) == 0
            tline2 = fgetl(curr_sum);
            if tline2 ~= -1 & size(tline2, 1) > 0
                tline1 = tline2;
            end
        end
        if tline1 == -1
            disp("End of File Reached")
        end
    end
end

% Once you have gone through all of the files, convert to a table and save
ref_table = struct2table(storms_sheet);
writetable(ref_table, "RT_MIT_Seizure_sheet.xlsx")
save("RT_MIT_storms_sheet.mat", 'storms_sheet')

        
