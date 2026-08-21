%% train_model.m
% Fits the detector in MATLAB and saves it for the live demonstration.
%
% Reads dataset_crop.csv, trains a bagged tree ensemble, picks a decision
% threshold on held-out data, and writes model.mat. demo_image.m and
% score_folder.m load that file, so the demo starts instantly instead of
% retraining every time.
%
% Bagged trees are used because they need no feature scaling - the 230 features
% span six orders of magnitude, from FFT ring energies in the hundreds of
% thousands down to correlations near 1 - and a tree split is invariant to that.
%
% Run with the project root as pwd, after feature_extractor.m.
%
% Requires: Statistics and Machine Learning Toolbox.

clear; clc;

CSV_PATH    = 'dataset_crop.csv';
NAMES_PATH  = 'filenames_crop.txt';
OUT_PATH    = 'model.mat';
N_TREES     = 150;
TEST_FRAC   = 0.20;
RANDOM_SEED = 42;

if ~isfile(CSV_PATH)
    error('train_model:missingCsv', ...
          'Cannot find %s - run feature_extractor.m first.', ...
          fullfile(pwd, CSV_PATH));
end

rng(RANDOM_SEED);

%% ------------------------------------------------------------------ load
data = readmatrix(CSV_PATH);
if size(data, 2) ~= 231
    error('train_model:badWidth', ...
          'Expected 231 columns, found %d.', size(data, 2));
end

X = data(:, 1:230);
y = data(:, 231);

fprintf('Loaded %d images x %d features (%d real / %d AI)\n', ...
        size(X,1), size(X,2), sum(y == 0), sum(y == 1));

%% -------------------------------------------------------- train / test
% cvpartition stratifies by class for classification, so both sides keep the
% same real/AI balance.
cv     = cvpartition(y, 'HoldOut', TEST_FRAC);
trIdx  = training(cv);
teIdx  = test(cv);

fprintf('Training bagged ensemble of %d trees on %d images...\n', ...
        N_TREES, sum(trIdx));

mdl = fitcensemble(X(trIdx,:), y(trIdx), ...
                   'Method', 'Bag', ...
                   'NumLearningCycles', N_TREES);

%% -------------------------------------------------------------- evaluate
% For a bagged ensemble the second score column is P(class = 1), because
% ClassNames comes back as [0;1] for a numeric 0/1 label.
[~, scores] = predict(mdl, X(teIdx,:));
prob  = scores(:, 2);
yTest = y(teIdx);

[~, ~, thresholds, aucValue] = perfcurve(yTest, prob, 1);
[rocX, rocY] = perfcurve(yTest, prob, 1);

% Youden's J: the operating point maximising balanced accuracy, rather than
% leaving the cutoff at an arbitrary 0.5.
[~, bestIdx] = max(rocY - rocX);
threshold    = thresholds(bestIdx);
if ~isfinite(threshold)
    threshold = 0.5;
end

accDefault = mean((prob >= 0.5)       == yTest);
accTuned   = mean((prob >= threshold) == yTest);

fprintf('\n  held-out ROC-AUC     : %.4f\n', aucValue);
fprintf('  accuracy @ 0.5       : %.4f\n', accDefault);
fprintf('  chosen threshold     : %.4f\n', threshold);
fprintf('  accuracy @ threshold : %.4f\n', accTuned);

%% ------------------------------------------- reference stats for the demo
% So demo_image.m can explain a verdict: for the most discriminative features,
% what a typical real and a typical AI image measure.
meanReal = mean(X(y == 0, :), 1);
meanAI   = mean(X(y == 1, :), 1);
pooled   = sqrt((var(X(y == 0, :), 0, 1) + var(X(y == 1, :), 0, 1)) / 2);
cohensD  = abs(meanReal - meanAI) ./ max(pooled, 1e-12);
cohensD(~isfinite(cohensD)) = 0;

[~, order]  = sort(cohensD, 'descend');
topFeatures = order(1:12);

fprintf('\nMost discriminative features (the demo uses these to explain itself):\n');
for i = 1:6
    c = topFeatures(i);
    fprintf('  col %3d  d=%.2f  real=%-10.4g AI=%-10.4g  %s\n', ...
            c, cohensD(c), meanReal(c), meanAI(c), describeFeature(c));
end

%% ------------------------------------------------------------------ save
generators = {};
if isfile(NAMES_PATH)
    generators = readGenerators(NAMES_PATH, y);
end

model = compact(mdl);        % smaller on disk, still supports predict
save(OUT_PATH, 'model', 'threshold', 'aucValue', 'accTuned', ...
     'meanReal', 'meanAI', 'cohensD', 'topFeatures', 'generators');

fprintf('\nSaved %s\n', fullfile(pwd, OUT_PATH));
fprintf('Next: demo_image  (live demonstration on one image)\n');
fprintf('      score_folder (test a folder of images from a new generator)\n');


%% ================================================================
%  Local functions
%  ================================================================

function gens = readGenerators(namesPath, y)
%READGENERATORS Generator tag from the 2nd underscore-separated filename field.
%   Parsed only for AI rows - a real filename such as
%   ILSVRC2012_val_00001277 would otherwise parse to 'val'.

    fid = fopen(namesPath, 'r');
    if fid == -1
        gens = {};
        return;
    end
    cleanup = onCleanup(@() fclose(fid));

    names = {};
    line  = fgetl(fid);
    while ischar(line)
        names{end+1} = line; %#ok<AGROW>
        line = fgetl(fid);
    end

    if numel(names) ~= numel(y)
        warning('train_model:rowMismatch', ...
                '%s has %d lines but the CSV has %d rows; skipping generators.', ...
                namesPath, numel(names), numel(y));
        gens = {};
        return;
    end

    found = {};
    for i = 1:numel(names)
        if y(i) ~= 1
            continue;
        end
        [~, base] = fileparts(strrep(names{i}, '\', '/'));
        parts = strsplit(base, '_');
        if numel(parts) >= 2
            found{end+1} = lower(parts{2}); %#ok<AGROW>
        end
    end
    gens = unique(found);
end
