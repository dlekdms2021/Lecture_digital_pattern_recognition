% ICA 오디오 분리 과제

% 혼합된 오디오 로드
[X, fs] = audioread('../data/mixed_audio_2mix_v2.wav');  % X: (samples x channels)

disp('Mixed audio dimension:');
disp(size(X));  % 예: [10625664 2]

% Play the given audio mix
% player = audioplayer(X, fs);
% play(player)

% Stop playing
% stop(player)

% ICA 실행을 위한 준비
X = X';  % Transpose to (channels x samples)


% Run ICA runica() in EEGLAB (it could be slow)
% ================= Your code =========================
[weights, sphere] = runica(X);  % ICA 수행
X_hat = weights * sphere * X;   % 분리된 신호 계산


% ================ Your code (refer to the lecture material)
% X_hat = icaweights * icasphere * X;
% Normalization
X_hat = X_hat ./ max(abs(X_hat), [], 2);

% 5. 원본 오디오 소스 불러오기
audio_dir = '../data/example_sound_sources/example_sound_sources/';
files = dir(fullfile(audio_dir, '*.wav'));

min_len = inf;
fs_all = zeros(length(files),1);
S = [];

for i = 1:length(files)
    [y, fs_tmp] = audioread(fullfile(audio_dir, files(i).name));
    y = y(:,1);  % mono 사용
    fs_all(i) = fs_tmp;
    min_len = min(min_len, length(y));
    S = [S; y'];  % (파일 수 x samples)
end

% 샘플링 레이트 통일성 체크
if ~all(fs_all == fs_all(1))
    error('All audio files must have the same sampling rate');
end
fs = fs_all(1);

disp('Dimension of original source list');
disp(size(S));  % 예: [11, N] 형태

% 6. 비교를 위한 다운샘플링
down_Xhat = resample(X_hat', 1, 48)';  % (2 x N')
down_S = resample(S', 1, 48)';        % (11 x N')

% 7. 추정된 소스와 원본 소스 비교 (상관계수)
for i = 1:size(S,1)
    r = corr(down_Xhat(1,:)', down_S(i,:)');
    fprintf('Xhat 1 vs Original %d: Corr = %.4f\n', i, r);
end

for i = 1:size(S,1)
    r = corr(down_Xhat(2,:)', down_S(i,:)');
    fprintf('Xhat 2 vs Original %d: Corr = %.4f\n', i, r);
end

% 8. 분리된 오디오 저장
for i = 1:size(X_hat, 1)
    fname = sprintf('../data/estimated_source%02d.wav', i);
    audiowrite(fname, X_hat(i, :)', fs);
end