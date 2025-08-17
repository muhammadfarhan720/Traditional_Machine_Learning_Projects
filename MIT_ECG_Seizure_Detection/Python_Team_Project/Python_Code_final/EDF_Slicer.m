%% EDF_Slicer.m
% Jeremy Decker
% 11/1/2022
% This script will take in seizure files from either the MIT scalp database
% for the Temple university EEG corpus, extract all seizures in 5s
% segments, and an equal number of segmenst of non-seizure data sampled
% from throughout the file, based on file length. 

%% Import a reference file, containing a list of files, and save dest. 
already_ref_file = exist('ref_file', 'var');
if already_ref_file == 0
    [fpath, foldpath] = uigetfile("", "Please select a file for summary file analysis");
    ref_path = strcat(foldpath,fpath);
    ref_file = readtable(ref_path);
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
        % 
        seizure_starts = [seizure_starts, s_index-10*srate];
        seizure_stops = [seizure_stops, e_index+10*srate];
        seizure_durations = [seizure_durations, duration];

        %% Get the Seizure Segments
        for k = 1:duration_epochs
            epoch_start = (k-1)*srate*5 + s_index;
            epoch_stop = k*srate*5 + s_index;
            seizure_seg = transpose(EEG.data(:, epoch_start:epoch_stop));
            seizure_table = array2table(seizure_seg, "VariableNames", chan_names);
            table_name = strcat("Siena\Seizures\", fname, "_S_", string(j),'seg_', string(k), '.csv');
            writetable(seizure_table, table_name)
        end
        %% Get the non-seizure epochs, but only once. 
        if j == length(file_list(i).entries)
            disp('Making Non-Seizure Segments')
            total_seizure_time = sum(seizure_durations);
            total_seizure_epochs = floor(total_seizure_time/5);
            % Collect all available starting indices for the non-seizure
            % segments
            for k = 1:length(seizure_starts)
                if k == 1
                    avail_inds = (60*srate):seizure_starts(k);
                else
                    avail_inds = [avail_inds, seizure_stops(k-1):seizure_starts(k)];
                end
                if k == length(seizure_starts)
                    avail_inds = [avail_inds, seizure_stops(k):(EEG.pnts-60*srate)];
                end
            end
            % From here, grab a random start index, then get the
            % non-seizure segments available. 
            for k = 1:total_seizure_epochs
                start_val = randi(length(avail_inds));
                % Since we have already established a buffer between
                % seizure starts and stops, we can simply grab the proper
                % indices for the work. 
                start_ind = avail_inds(start_val);
                stop_ind = start_ind + 5*srate;
                end_val = start_val + 5*srate;
                % This will be used to overzealously remove available
                % indices to ensure a gap between non-storm samples. 
                if end_val > length(avail_inds)
                    end_val = length(avail_inds);
                end
                % Eliminate the segment from the running, plus a buffer to
                % avoid overlaps. 
                avail_inds = avail_inds([1:(start_val-5*srate), (end_val):length(avail_inds)]);
                ns_seg = transpose(EEG.data(:, start_ind:stop_ind));
                ns_table = array2table(ns_seg, "VariableNames", chan_names);
                table_name = strcat("Siena\NS_Segs\", fname, "_NS_", string(j),'seg_', string(k), '.csv');
                writetable(ns_table, table_name)
            end
        end
    end
end