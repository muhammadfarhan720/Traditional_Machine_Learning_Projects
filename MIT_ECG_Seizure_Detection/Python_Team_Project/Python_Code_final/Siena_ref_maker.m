% Sienna_ref_maker.m
% Jeremy Decker
% 11/10/2022
% This script, similar to the MIT_ref_maker.m script, will parse summary
% files from the Sienna scalp EEG databse and will break them up into
% seizure and non-seizure segments, for processing later in the
% EDF_Slicer.m file. 
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
        if regexpi(tline1, "File Name")
            if exist('curr_filename', 'var')
                if ~strcmp(curr_file,curr_filename)
                    disp("Found New Filename")
                    curr_filename = tline1(1, 12:length(tline1));
                    disp(curr_filename)
                    % Grab the next line to get start time
                    tline1 = fgetl(curr_sum);
                    file_start = regexp(tline1, '\d*[:.]\d*[:.]\d*', 'match');
                    if regexp(string(file_start), ":")
                        % Change any periods to : in the thing. 
                        file_start = strrep(file_start, '.', ':');
                        filestart = datetime(file_start);
                    else
                        filestart = datetime(file_start, 'InputFormat','HH.mm.ss');
                    end
                else
                    disp("Seizure recorded on same file, maintaining values")
                end
                disp("Check")
            else
                disp("Found New Filename")
                curr_filename = tline1(1, 12:length(tline1));
                disp(curr_filename)
                % Grab the next line to get start time
                tline1 = fgetl(curr_sum);
                file_start = regexp(tline1, '\d*[:.]\d*[:.]\d*', 'match');
                if regexp(string(file_start), ":")
                    % Change any periods to : in the thing. 
                    file_start = strrep(file_start, '.', ':');
                    filestart = datetime(file_start);
                else
                    filestart = datetime(file_start, 'InputFormat','HH.mm.ss');
                end
            end
        end
        
        if size(regexpi(tline1, "start time", "match"), 1) 
                disp("Passed Start Time Check")
                disp(regexpi(tline1, "registration"))
            if size(regexpi(tline1, "registration", 'match'), 1) == 0
                % Seizure found for the file, add the start and end times
                mes = sprintf("Seizure Found! Total Seizures %i, File Seizures: %i", total_seizures, file_seizures);
                disp(mes)
                disp(tline1)
                % Use a regular expression to pull the numbers out of the line
                start_time = regexp(tline1, '\d*[:.]\d*[:.]\d*', 'match');
                % Get the next line of the file
                tline1 = fgetl(curr_sum);
                stop_time = regexp(tline1, '\d*[:.]\d*[:.]\d*', 'match');
                % Conver the time values into datetime objects
                if regexp(string(start_time), ':')
                    start_time = strrep(start_time, '.', ':');
                    start_time = datetime(start_time);
                    if hour(start_time) < 12 && hour(filestart) >= 12
                        start_time = start_time + days(1);
                    end
                else
                    start_time = datetime(start_time, 'InputFormat','HH.mm.ss');
                    if hour(start_time) < 12 && hour(filestart) >= 12
                        start_time = start_time + days(1);
                    end
                end
                if regexp(string(stop_time), ':')
                    stop_time = strrep(stop_time, '.', ':');
                    stop_time = datetime(start_time);
                    if hour(stop_time) < 12 && hour(filestart) >= 12
                        stop_time = stop_time + days(1);
                    end
                else
                    stop_time = datetime(stop_time, 'InputFormat','HH.mm.ss');
                    if hour(stop_time) < 12 && hour(filestart) >= 12
                        stop_time = stop_time + days(1);
                    end
                end
                % Convert them to integers/doubles?
                start_elapsed = seconds(start_time-filestart);
                duration = seconds(stop_time-start_time);
%                 cf_exist = exist('curr_filename', 'var');
%                 if cf_exist
%                     if strcmp(curr_filename, "PN10-4.5.6.edf")
%                         disp("Checkpoint")
%                     end
%                 end
                
                disp("Adding storm to structure")
                storms_sheet(total_seizures).eeg = strcat(curr_path, '/', curr_filename);
                storms_sheet(total_seizures).start = start_elapsed;
                storms_sheet(total_seizures).duration = duration;
                storms_sheet(total_seizures).sum_file_seizures = file_seizures;
                storms_sheet(total_seizures).sum_file = curr_file;
                total_seizures = total_seizures+1;
                file_seizures = file_seizures+1;
            end
        end
        % Get the next line in the file
        tline1 = fgetl(curr_sum);
        if size(tline1, 1) == 0
            exit_term =0; 
            while exit_term ==0
                tlinetest = fgetl(curr_sum);
                if tlinetest ~= -1 & size(tlinetest, 1) >0
                    tline1 = tlinetest;
                    exit_term = 1;
                elseif tlinetest == -1
                    disp("End of file, exiting loop")
                    tline1 = tlinetest;
                    exit_term = 1;
                end
            end
        end
        if tline1 == -1
            disp("End of File Reached")
        end
    end
end

% Once you have gone through all of the files, convert to a table and save
ref_table = struct2table(storms_sheet);
writetable(ref_table, "RT_Siena_Seizure_sheet.xlsx")
save("RT_Siena_storms_sheet.mat", 'storms_sheet')

        
