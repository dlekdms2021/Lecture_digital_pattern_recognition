% Load mixed audio
[X, fs] = audioread('../data/mixed_audio_2mix_v2.wav');  % X: (samples x channels)
disp('Mixed audio dimension:');
disp(size(X));  % e.g., 10625664 x 2

% Play the given audio mix
% player = audioplayer(X, fs);
% play(player)
% stop(player)

% Transpose to (channels x samples)
X = X';

% Run ICA
[icaweights, icasphere] = runica(X);  % ICA 수행
X_hat = icaweights * icasphere * X;   % 원천 추정
X_hat = X_hat ./ max(abs(X_hat), [], 2);  % 정규화

% Play estimated sources
% player_estimated1 = audioplayer(X_hat(1, :), fs); play(player_estimated1); stop(player_estimated1);
% player_estimated2 = audioplayer(X_hat(2, :), fs); play(player_estimated2); stop(player_estimated2);

% Load original sources
audio_dir = '../data/example_sound_sources/example_sound_sources/';
files = dir(fullfile(audio_dir, '*.wav'));
min_len = inf;
fs_all = zeros(length(files),1);
S = [];
for i = 1:length(files)
    [y, fs_tmp] = audioread(fullfile(audio_dir, files(i).name));
    y = y(:,1);  % mono
    fs_all(i) = fs_tmp;
    min_len = min(min_len, length(y));
    S = [S; y'];
end
if ~all(fs_all == fs_all(1)), error('Sampling rates differ'); end
fs = fs_all(1);
disp('Dimension of original source list');
disp(size(S));  % e.g., 11 x samples

% Downsampling for correlation and visualization
down_xhat = resample(X_hat', 1, 48)';
down_S = resample(S', 1, 48)';

% Crop to equal length for fair comparison
min_len = min(size(down_xhat, 2), size(down_S, 2));
down_xhat = down_xhat(:, 1:min_len);
down_S = down_S(:, 1:min_len);

% Compare source 1 and 2 vs original sources
correlations = zeros(2, size(down_S,1));
distances = zeros(2, size(down_S,1));

for i = 1:size(down_S, 1)
    correlations(1,i) = corr(down_xhat(1,:)', down_S(i,:)');
    correlations(2,i) = corr(down_xhat(2,:)', down_S(i,:)');
    distances(1,i) = norm(down_xhat(1,:) - down_S(i,:));
    distances(2,i) = norm(down_xhat(2,:) - down_S(i,:));
end

% 결과 출력
disp('Correlation (Estimated Source 1 vs Original Sources):');
disp(correlations(1,:));
disp('Correlation (Estimated Source 2 vs Original Sources):');
disp(correlations(2,:));

disp('Distance (Estimated Source 1 vs Original Sources):');
disp(distances(1,:));
disp('Distance (Estimated Source 2 vs Original Sources):');
disp(distances(2,:));

% 스펙트로그램 시각화 (추정 source vs 상위 상관 원본 source)
[~, idx1] = max(correlations(1,:));
[~, idx2] = max(correlations(2,:));

figure;
subplot(2,2,1); spectrogram(down_xhat(1,:), 256, [], [], fs/48, 'yaxis'); title('Estimated Source 1');
subplot(2,2,2); spectrogram(down_S(idx1,:), 256, [], [], fs/48, 'yaxis'); title(sprintf('Best Match Source %d', idx1));
subplot(2,2,3); spectrogram(down_xhat(2,:), 256, [], [], fs/48, 'yaxis'); title('Estimated Source 2');
subplot(2,2,4); spectrogram(down_S(idx2,:), 256, [], [], fs/48, 'yaxis'); title(sprintf('Best Match Source %d', idx2));

% 결과 저장
for i=1:size(X_hat, 1)
    fname = sprintf('../data/estimated_source%02d.wav', i);
    audiowrite(fname, X_hat(i, :)', fs);
end
