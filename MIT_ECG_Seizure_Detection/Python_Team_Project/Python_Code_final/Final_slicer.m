%% Final_Slicer.m
% Jeremy Decker
% 12/3/2022
% This script will take in a reference sheet containign the remaining EDF
% files set aside and convert them into 5s chunks, labeled as seizures if
% they have any time with seizures in them, and labeled as non-seizures
% otherwise. 

%% Import a reference file, containing a list of files, and save dest. 
already_ref_file = exist('ref_file', 'var');
if already_ref_file == 0
    [fpath, foldpath] = uigetfile("", "Please select a file for summary file analysis");
    ref_path = strcat(foldpath,fpath);
    ref_file = readtable(ref_path);
end
% Determine the dataset name for saving purposes
if regexpi(fpath, 'MIT')
    base_name = "MIT";
else
    base_name = "Siena";
end
%% Create a files list
numfiles = 1;
tracked_files = [];
eeglab;
for i = 1:length(ref_file.eeg)
    if i == 1
        file_list(numfiles).name = ref_file.eeg(i);
        file_list(numfiles).entries = i;
        tracked_files = ref_file.eeg(i);
    elseif any(strcmp(ref_file.eeg(i), tracked_files))
        for j = 1:length(tracked_files)
            if strcmp(ref_file.eeg(i), tracked_files(j))
                file_list(j).entries = [file_list(j).entries, i];
            end
        end
    else
        numfiles = numfiles+1;
        file_list(numfiles).name = ref_file.eeg(i);
        file_list(numfiles).entries = i;
        tracked_files = [tracked_files, ref_file.eeg(i)];
    end
end
%% Start a loop to go through each file
for i = 1:length(file_list)
    curr_file = string(file_list(i).name);
    mes = sprintf("Processing file: %s", curr_file);
    disp(mes)
    [path, fname, ext] = fileparts(curr_file);
    EEG = pop_biosig(curr_file);
    srate = EEG.srate;
    seizure_time = 0;
    chan_names = strings(length(EEG.chanlocs),1);
    for j = 1:length(EEG.chanlocs)
        if any(strcmp(EEG.chanlocs(j).labels, chan_names))
            temp_name = strcat(EEG.chanlocs(j).labels,'-2');
            if any(strcmp(temp_name, chan_names))
                matched = 0; 
                test_num = 3;
                while matched == 0
                    test_str = strcat(EEG.chanlocs(j).labels, test_num);
                    if ~any(strcmp(test_str, chan_names))
                        chan_names(j) = test_str;
                        matched = 1;
                    else
                        test_num = test_num+1;
                    end
                end
            else
                chan_names(j) = temp_name;
            end
        else
            chan_names(j)= strcat(EEG.chanlocs(j).labels);
        end
    end
    %% Identify seizures using the provided sheet
    seizures_length = 0;
    seizure_starts = [];
    seizure_stops = [];
    seizure_durations = [];
    for j = 1:length(file_list(i).entries)
        % Fetch the start time and duration of the storm
        start_time = ref_file.start(file_list(i).entries(j));
        duration = ref_file.duration(file_list(i).entries(j));
        duration_epochs = floor(duration/5); % Get the number of 5 second segments to be made, conservatively
        % Get the starting and ending index of the storm.
        s_index = start_time*srate+1;
        d_index = s_index + duration_epochs*srate*5; % convert to indices
        e_index = s_index + duration*srate;
        % Grab the start and end indices with a buffer to prevent overlaps.
        seizure_starts = [seizure_starts, s_index];
        seizure_stops = [seizure_stops, e_index];
    end
    % Find the number of 5 second chunks you can make from the file
    total_segments = floor(EEG.pnts/(5*srate));
    %% Create Sorted Chunks based on the seizure times and start and stop inds
    curr_seizure = 1;
    for j = 0:(total_segments-1)
        bot_ind = 1 + j*5*srate;
        top_ind = (j+1)*5*srate;
        % Check for seizures
        if bot_ind < seizure_starts(curr_seizure) && seizure_starts(curr_seizure) < top_ind && top_ind < seizure_stops(curr_seizure)
            % Case 1: Segment Enters Seizure
            seizure_status = 1;
        elseif bot_ind > seizure_starts(curr_seizure) && bot_ind < seizure_stops(curr_seizure)
            % Case 2: Segment Starts in Seizure
            seizure_status = 1;
        elseif bot_ind < seizure_starts(curr_seizure) && top_ind > seizure_stops(curr_seizure)
            % Case 3: Seizure Encapsulated by Segment- Theoretically
            % shouldn't happen, but I am covering my bases. 
            seizure_status = 1;
        else
            % Case 4: Segment is not in a seizure at all
            seizure_status = 0;
        end
        
        % Collect the segment
        seizure_seg = transpose(EEG.data(:, bot_ind:top_ind));
        seizure_table = array2table(seizure_seg, "VariableNames", chan_names);
        % Save the segment, giving it a savestring based on seizure_status
        % In this case, the segment number is the raw segment number, not
        % the seizure segment or comparative NS segment. 
        % This will save it to one large folder, as the prediction analysis
        % will happen on this whole dataset, not the training or testing
        % data. 
        if seizure_status == 1
            save_string = strcat(base_name, "_Final\Seizures\", fname, "_S_", string(curr_seizure),'seg_', string(j), '.csv');
        else
            save_string = strcat(base_name, "_Final\NS_Segs\", fname, "_NS_", string(curr_seizure),'seg_', string(j), '.csv');
        end
        % Save the File
        writetable(seizure_table, save_string)
        % Now, Check to see if we need to move to the next seizure
        if curr_seizure < length(seizure_starts)
            if top_ind >= seizure_stops(curr_seizure)
                % If the end of the current segment exceeds the end of the
                % seizure, move to the next one
                curr_seizure = curr_seizure+1;
            end
        end
    end
end