%%

eeglab;
ft_defaults;
add_paths_Won2021;

%%
% P300Speller Visualization ver1
%
% Kyungho Won
%
% [ reference ]
% data: BCI2000, 32 Biosemi2, 55 subjects
% target text: BRAIN, POWER / SUBJECT, NEURONS, IMAGINE, QUALITY
% target : standard = 30 : 150
%
% [ Stage ]
%  1. Pre-processing: bandpass filtering, extracting triggers
%     - freq = [0.5 40]
%     - frame = [0 1000]
%     - baseline = [-200 0]

clear; clc;
ch = 1:32; % select channels

Params_P3speller = struct('freq', [1 40], 'frame', [0 1000], ...
    'baseline', [-200 0], 'select_ch', 1:32);
electrodes_midline = {'FZ', 'Cz', 'Pz'};
electrodes_eyes = {'FP1', 'FP2', 'AF3', 'AF4'};

for nsb=1
    fname_train = sprintf('../data/s%02d.mat', nsb);
    EEG = load(fname_train);
    eeg_test = EEG.test;
    
    % ------------------------ CALIBRATION EEG ------------------------- %
    eeg_target = [];
    eeg_nontarget = [];
    for nRun = 1:length(eeg_test)
        cur_eeg = eeg_test{nRun};
        interest_ch = ismember({cur_eeg.chanlocs.labels}, electrodes_midline);
        [cur_target, cur_nontarget] = preproc_extractEpoch(cur_eeg, Params_P3speller);
        
        eeg_target = cat(3, eeg_target, cur_target);
        eeg_nontarget = cat(3, eeg_nontarget, cur_nontarget);
    end
    
    figure,
    t = linspace(Params_P3speller.baseline(1), Params_P3speller.frame(2), size(cur_target, 2));
    avg_target = mean(eeg_target, 3)';
    avg_nontarget = mean(eeg_nontarget, 3)';
    
    std_target = std(cur_target, [], 3)';
    std_nontarget = std(cur_nontarget, [], 3)';
    vis_ERP(t, mean(avg_target(:, interest_ch),2), mean(avg_nontarget(:, interest_ch),2), ...
        Params_P3speller.baseline, 0:200:1000, std_target(:, interest_ch), std_nontarget(:, interest_ch), 'off');
    
    topo3D = cat(3, avg_target', avg_nontarget', (avg_target-avg_nontarget)');
    clim = [-3 3];
    frames = 0:200:1200;
    figure,
    vis_temporalTopoplot(topo3D, cur_eeg.srate, frames, cur_eeg.chanlocs, clim);
    colormap(redblue);
end


%%
% Influence of noisy period

close all;
% data before additional filtering (bandpass filter applied during recording - 0.1
% to 70 Hz + notch filtering at 60 Hz
data_untouched = cur_eeg.data;
srate = cur_eeg.srate;
ch_locs = cur_eeg.chanlocs; % channel location info

figure,
spectopo(data_untouched, 0, srate);
xlim([0, 70]);

eegplot(data_untouched, 'srate', srate, 'winlength', 15, 'eloc_file', ch_locs);

% Crop data containing huge distortion to check contribution of artifacts
% to PSD
data_near_artifacts = data_untouched(:, 86*srate:115*srate);
eegplot(data_near_artifacts, 'srate', srate, 'winlength', 15, 'eloc_file', ch_locs);
t = linspace(86, 115, length(data_near_artifacts));
figure,
subplot(2,1,1);
plot(t, data_near_artifacts'); title('Butterfly plot');
xlabel('Time (s)'); ylabel('\muV'); pbaspect([1, 1, 1]);

subplot(2,1,2);
spectopo(data_near_artifacts, 0, srate); pbaspect([1, 1, 1]);
xlim([0, 70]);
sgtitle('Including noisy period', 'fontsize', 16);   
set(gcf, "Position", [300, 300, 1024, 486]);

% Exclude noisy period
data_near_artifacts(:, 9*srate:21*srate) = [];
eegplot(data_near_artifacts, 'srate', srate, 'winlength', 15, 'eloc_file', ch_locs);
figure,
subplot(2, 1,1);
plot(data_near_artifacts'); title('Butterfly plot');
xlabel('Time (s)'); ylabel('\muV'); pbaspect([1, 1, 1]);

subplot(2, 1,2);
spectopo(data_near_artifacts, 0, srate); pbaspect([1, 1, 1]);
xlim([0, 70]); pbaspect([1, 1, 1]);
sgtitle('Excluding noisy period');
set(gcf, "Position", [300, 300, 1024, 486]);

%% Test phase shift using FIR filter

% Example: low-pass FIR filter (causal)
Fs = srate;  % Sampling rate
fc = 30;   % Cut-off frequency
order = 100;

b = fir1(order, fc/(Fs/2));  % FIR filter coefficients
figure, 
freqz(b, 1, Fs); % filter property

eeg_shifted = filter(b, 1, data_untouched');  % Apply causal filter
eegplot(data_untouched, 'data2', eeg_shifted', 'srate', cur_eeg.srate, 'eloc_file', cur_eeg.chanlocs);

eeg_zero_phase_filtered = filtfilt(b, 1, data_untouched');  % Apply causal filter
eegplot(data_untouched, 'data2', eeg_zero_phase_filtered', ...
    'srate', cur_eeg.srate, 'eloc_file', cur_eeg.chanlocs);


%% Data segmentation
disp(Params_P3speller);
event_markers = cur_eeg.markers_target;
disp(unique(event_markers));

tmp_target = event_markers;
tmp_target(tmp_target==2) = 0; % remove non-target markers
% tmp_target: 1 for target o/w 0

tmp_nontarget = event_markers;
tmp_nontarget(tmp_nontarget==1) = 0; % remove target merkers
tmp_nontarget = sign(tmp_nontarget);

wn = Params_P3speller.freq / (srate/2);
[b, a] = butter(4, wn, 'bandpass');
% demean the data before filtering
meandat = mean(data_untouched, 2);
data_untouched = bsxfun(@minus, data_untouched, meandat);
filt_eeg = filtfilt(b, a, data_untouched')';


epoch_target = eeg_extract_epochs(filt_eeg, srate, tmp_target, Params_P3speller.frame);
epoch_nontarget = eeg_extract_epochs(filt_eeg, srate, tmp_nontarget, Params_P3speller.frame);

disp('Target epochs [ch x time x samples]');
disp(size(epoch_target));

disp('Nontarget epochs [ch x time x samples');
disp(size((epoch_nontarget)));

eegplot(epoch_target, 'srate', srate, 'eloc_file', ch_locs, 'title', 'target', 'winlength', 13);
eegplot(epoch_nontarget, 'srate', srate, 'eloc_file', ch_locs, 'title', 'nontarget', 'winlength', 13);

%% Run SVD and PCA

[W, Y] = pca(filt_eeg);
figure,
for i=1:size(filt_eeg, 1)-1
    subplot(6,6,i);
    topoplot(W(i, :), ch_locs);
end
eegplot(filt_eeg, 'srate', srate, 'eloc_file', ch_locs, 'title', 'target', 'winlength', 13);
eegplot(Y, 'srate', srate, 'eloc_file', ch_locs);


% PCA: covariance matrix
C = cov(filt_eeg');
[Vpca, D] = eig(C);              
[~, idx] = sort(diag(D), 'descend');
Vpca = Vpca(:, idx);    
D = D(idx, idx);           

% SVD: filtered data
[U, S, V] = svd(filt_eeg, 'econ');        % X = U * S * V'

% First principal component
figure;
plot(Vpca(:,1), '-o');
title('1st Principal Component (PCA)');
xlabel('Channel Index');
ylabel('Weight');

% SVD
figure;
plot(U(:,1), '-s');
title('1st Spatial Component (SVD)');
xlabel('Channel Index');
ylabel('Weight');

% Select components
k = 5;
X_reduced = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';  % 근사 복원

% Origin vs. reconstructed
eegplot(filt_eeg, 'srate', srate, 'eloc_file', ch_locs, 'data2', X_reduced);


%% Run independent component analysis

disp('Artifact removal using ICA ...');
% Y = Ax
ica = [];
[ica.weights, ica.sphere] = runica(filt_eeg, 'extended', 1);
ica.unmix = ica.weights * ica.sphere;
ica.winv = inv(ica.unmix); % A hat (weight inverse)
ica.ics = ica.unmix * filt_eeg;
% estimated sources: ica.ics = ica.weights * ica.sphere * X

eegplot(ica.ics, 'srate', srate, 'winlength', 15);
figure,
for i=1:size(filt_eeg, 1)
    subplot(6,6,i);
    topoplot(ica.winv(:,i),ch_locs); colorbar;...
        title(strcat('Components ', num2str(i))); colormap('jet')
end

figure,
subplot(2,1,1); spectopo(filt_eeg, 0, srate, 'overlap', srate/2);
xlim([0 70]); ylim([-20 40]); title('raw EEG spectral power -- spectopo()');

subplot(2,1,2); spectopo(ica.ics, 0, srate, 'overlap', srate/2);
xlim([0 70]); ylim([-20 40]); title('Components spectral power -- spectopo()');

%%

reject_set = input('Please type \nEEG ICA componets that you want to reject: ', 's');
reject_idx = str2num(reject_set);

ica.mask_winv = ica.icawinv;
ica.mask_winv(:, reject_idx) = 0;
clean_data = ica.mask_winv * ica.ics;

eegplot(filt_eeg, 'data2', clean_data, 'srate', srate, 'winlength', 10, ...
    'title', 'Black = channel before rejection; red = after rejection -- eegplot()', ...
    'eloc_file', ch_locs);

figure,
subplot(2,1,1);
spectopo(filt_eeg, 0, srate); xlim([0 40]);
title('Before component rejection -- spectopo()'); pbaspect([1, 1, 1]);

subplot(2,1,2);
spectopo(clean_data, 0, srate); xlim([0 40]);
title('After component rejection -- spectopo()'); pbaspect([1, 1, 1]);
EEG.reject_ica_comp = reject_idx;

%% sub functions

function out = eeg_extract_epochs(data2D, srate, event_markers, frame)

out = [];
ind = find(event_markers > 0);
for nTrials=1:length(ind)
    begin_frame = ind(nTrials) + floor(frame(1)/1000*srate);
    end_frame = ind(nTrials) + floor(frame(2)/1000*srate)-1;
    out = cat(3, out, data2D(:, begin_frame:end_frame));
end

end