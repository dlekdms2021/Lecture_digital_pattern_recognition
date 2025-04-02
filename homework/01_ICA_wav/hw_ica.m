%% 초기 설정
eeglab;        % EEGLAB 툴박스 초기화 (runica 함수 사용을 위해 필요)
clear; clc;    % 변수와 콘솔 초기화

%% 혼합 오디오 불러오기
[X, fs] = audioread('./data/mixed_audio_2mix_v2.wav');  % 혼합된 오디오 (샘플 수 × 채널 수)
fprintf('Mixed audio shape: [%d samples × %d channels]\n', size(X,1), size(X,2));
X = X';  % ICA 처리를 위해 (채널 수 × 샘플 수) 형태로 전치

%% ICA 수행
fprintf('Running ICA...\n');
[weights, sphere] = runica(X);                 % ICA 수행 → 분리 행렬(weights), 구면화 행렬(sphere)
X_hat = weights * sphere * X;                  % 원천 성분 추정 (ICA 분리된 소스)
X_hat = X_hat ./ max(abs(X_hat), [], 2);       % 정규화 (최대값 기준으로 [-1,1] 범위)

%% 원본 소스 불러오기
audio_dir = './data/example_sound_sources/example_sound_sources/';
files = dir(fullfile(audio_dir, '*.wav'));     % 예제 원본 소스들 불러오기

min_len = inf;                                 % 소스들 중 가장 짧은 길이를 찾기 위한 초기값
S = [];                                        % 원본 소스들을 저장할 행렬
fs_all = zeros(length(files),1);               % 샘플링 주파수 일치 여부 확인용

for i = 1:length(files)
    [y, fsi] = audioread(fullfile(audio_dir, files(i).name));  % 파일 불러오기
    y = y(:,1);                                 % mono 채널만 사용
    fs_all(i) = fsi;                            % 샘플링 주파수 저장
    min_len = min(min_len, length(y));          % 가장 짧은 소스 길이 찾기
    S = [S; y'];                                % 행 방향으로 소스를 쌓음 (source index × 샘플 수)
end

if ~all(fs_all == fs_all(1))
    error('Sampling rates are not consistent across files.');  % 모든 소스의 샘플링 주파수가 같아야 함
end
fs = fs_all(1);  % 공통 샘플링 주파수 설정

%% 길이 정렬 및 다운샘플링
min_len_resample = min(size(X_hat, 2), size(S, 2));  % 분리된 소스와 원본 소스 중 짧은 길이
X_hat = X_hat(:, 1:min_len_resample);                % 길이 맞춰 자르기
S = S(:, 1:min_len_resample);                        % 길이 맞춰 자르기

down_Xhat = resample(X_hat', 1, 48)';  % 다운샘플링 (샘플 수 줄이기, 비교/시각화 효율)
down_S = resample(S', 1, 48)';

min_len_corr = min(size(down_Xhat, 2), size(down_S, 2));  % 다운샘플 후 길이 재확인
down_Xhat = down_Xhat(:, 1:min_len_corr);                 % 다시 길이 맞춤
down_S = down_S(:, 1:min_len_corr);

%% 상관계수 및 거리 계산
corr_mat = corr(down_Xhat', down_S');     % 상관계수 행렬 (2 x N_sources)
distances = zeros(2, size(down_S, 1));    % 거리(유클리드 norm) 저장할 행렬

for i = 1:size(down_S, 1)
    distances(1,i) = norm(down_Xhat(1,:) - down_S(i,:));  % 추정 소스 1과 원본 소스 i 거리
    distances(2,i) = norm(down_Xhat(2,:) - down_S(i,:));  % 추정 소스 2와 원본 소스 i 거리
end

%% 결과 출력
fprintf('\n--- Correlation (상관계수) ---\n');
disp(corr_mat);  % 추정된 소스들과 원본 소스 간의 상관계수 출력

% 상관계수가 가장 높은 원본 소스 인덱스 추출
fprintf('Estimated source 1 best matches with original source: %d\n', find(corr_mat(1,:) == max(corr_mat(1,:))));
fprintf('Estimated source 2 best matches with original source: %d\n', find(corr_mat(2,:) == max(corr_mat(2,:))));

fprintf('\n--- Distance (유클리드 거리) ---\n');
disp(distances);  % 추정된 소스들과 원본 소스 간의 거리 출력

%% 시각화: 상관계수 히트맵
figure;
imagesc(corr_mat);                    % 상관계수 행렬 시각화 (heatmap)
xlabel('Original Source Index'); 
ylabel('Estimated Source Index');
colorbar; 
title('Correlation Heatmap');

%% 시각화: 스펙트로그램 비교
[~, idx1] = max(corr_mat(1,:));       % 소스 1과 가장 유사한 원본 소스 인덱스
[~, idx2] = max(corr_mat(2,:));       % 소스 2와 가장 유사한 원본 소스 인덱스

figure;
subplot(2,2,1); spectrogram(down_Xhat(1,:), 256, [], [], fs/48, 'yaxis'); title('Estimated Source 1');
subplot(2,2,2); spectrogram(down_S(idx1,:), 256, [], [], fs/48, 'yaxis'); title(sprintf('Matched Original Source %d', idx1));
subplot(2,2,3); spectrogram(down_Xhat(2,:), 256, [], [], fs/48, 'yaxis'); title('Estimated Source 2');
subplot(2,2,4); spectrogram(down_S(idx2,:), 256, [], [], fs/48, 'yaxis'); title(sprintf('Matched Original Source %d', idx2));
sgtitle('Spectrogram Comparison between Estimated and Original Sources');

%% 오디오 파일 저장
for i = 1:size(X_hat, 1)
    fname = sprintf('./data/estimated_source%02d.wav', i);  % 저장 파일명 생성
    audiowrite(fname, X_hat(i,:)', fs);                     % 분리된 오디오 저장
    fprintf('Saved: %s\n', fname);                           % 저장 알림 출력
end
