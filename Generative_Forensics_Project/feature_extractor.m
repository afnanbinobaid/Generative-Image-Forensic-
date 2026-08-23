%% feature_extractor.m
% Batch DSP feature extraction for AI-generated image forensics.
%
% Walks Dataset/Real_Images (label 0) and Dataset/AI_Images (label 1), extracts
% a fixed 1x230 feature vector per image via extractImageFeatures(), appends the
% label, and writes dataset_<mode>.csv (no header, pure numeric) alongside a
% row-aligned filenames_<mode>.txt.
%
% The feature definitions live in extractImageFeatures.m, which demo_image.m
% also calls - so a live demonstration measures images exactly the same way this
% batch run did.
%
% Feature layout (230 features + 1 label = 231 columns):
%   Block A  cols   1 - 51    Global spatial stats, per channel R,G,B (17 x 3)
%   Block B  cols  52 - 198   2-level db4 DWT subband stats, per channel (49 x 3)
%   Block C  cols 199 - 218   FFT radial power spectrum, luminance (20)
%   Block D  cols 219 - 227   High-pass noise residual stats, luminance (9)
%   Block E  cols 228 - 230   Cross-channel correlation R-G, R-B, G-B (3)
%   Label    col  231         0 = Real, 1 = AI
%
% Run with the project root (the folder containing Dataset/) as pwd.
%
% Requires: Image Processing, Wavelet, and Statistics & Machine Learning
% Toolboxes.

clear; clc;

%% ---------------------------------------------------------------- config
PREPROCESS    = 'crop';      % 'crop'   = cut a 256x256 window from the centre
                             %            at native scale. Pixels are copied
                             %            untouched - no resampling.
                             % 'resize' = scale the whole frame to 256x256,
                             %            averaging pixels together, which
                             %            low-pass filters the image.
                             % See standardiseSize() in extractImageFeatures.m.
NUM_FEATURES  = 230;         % feature count, excluding the label
PROGRESS_STEP = 100;         % print a progress line every N images

AUG_SAMPLE_FRAC = 0.5;       % fraction of AUGMENTED siblings to extract.
                             % Every original image is always kept in full;
                             % this only thins the _qhi/_qlo/_rweb (or legacy
                             % _q85/_q60/_r75q85) copies make_augmented.m adds.
                             % 0.5 roughly halves extraction time on an
                             % already-augmented dataset without dropping any
                             % original photograph. 1.0 = extract everything.
AUG_SAMPLE_SEED = 42;        % reproducible sampling

projectRoot = pwd;
realDir     = fullfile(projectRoot, 'Dataset', 'Real_Images');
aiDir       = fullfile(projectRoot, 'Dataset', 'AI_Images');
csvPath     = fullfile(projectRoot, sprintf('dataset_%s.csv', PREPROCESS));
namesPath   = fullfile(projectRoot, sprintf('filenames_%s.txt', PREPROCESS));

if ~isfolder(realDir)
    error('feature_extractor:missingFolder', 'Cannot find %s', realDir);
end
if ~isfolder(aiDir)
    error('feature_extractor:missingFolder', 'Cannot find %s', aiDir);
end

%% ------------------------------------------------------------- file list
realFiles = listImages(realDir);
aiFiles   = listImages(aiDir);

if AUG_SAMPLE_FRAC < 1
    [realFiles, nRealDropped] = sampleAugmented(realFiles, AUG_SAMPLE_FRAC, AUG_SAMPLE_SEED);
    [aiFiles,   nAiDropped]   = sampleAugmented(aiFiles,   AUG_SAMPLE_FRAC, AUG_SAMPLE_SEED);
    fprintf('AUG_SAMPLE_FRAC %.2f: dropped %d augmented Real siblings, %d augmented AI siblings\n', ...
            AUG_SAMPLE_FRAC, nRealDropped, nAiDropped);
    fprintf('(every original image is kept; only _qhi/_qlo/_rweb copies were thinned)\n\n');
end

% Concatenate both classes up front so progress reporting reflects the
% combined total rather than restarting per folder.
allFiles  = [realFiles;                aiFiles];
allLabels = [zeros(numel(realFiles),1); ones(numel(aiFiles),1)];

totalImages = numel(allFiles);
if totalImages == 0
    error('feature_extractor:noImages', ...
          'No images found under %s or %s', realDir, aiDir);
end

fprintf('Found %d images (%d real, %d AI).\n', ...
        totalImages, numel(realFiles), numel(aiFiles));
fprintf('Preprocessing mode: %s\n', PREPROCESS);
fprintf('Extracting %d features per image...\n\n', NUM_FEATURES);

%% ------------------------------------------------------------ main loop
% Pre-allocate the full matrix; trim to the processed rows at the end.
featureMatrix  = NaN(totalImages, NUM_FEATURES + 1);
processedPaths = cell(totalImages, 1);

nProcessed = 0;
nSkipped   = 0;

startTime = tic;

for k = 1:totalImages
    imgPath = allFiles{k};

    try
        featureVector = extractImageFeatures(imgPath, PREPROCESS);

        nProcessed = nProcessed + 1;
        featureMatrix(nProcessed, :) = [featureVector, allLabels(k)];
        processedPaths{nProcessed}   = imgPath;

    catch ME
        nSkipped = nSkipped + 1;
        fprintf(2, 'SKIPPED: %s\n         %s\n', imgPath, ME.message);
        continue;
    end

    if mod(k, PROGRESS_STEP) == 0
        fprintf('Processed %d / %d images (%d skipped)\n', ...
                k, totalImages, nSkipped);
    end
end

elapsed = toc(startTime);

%% ---------------------------------------------------------------- write
featureMatrix  = featureMatrix(1:nProcessed, :);
processedPaths = processedPaths(1:nProcessed);

writematrix(featureMatrix, csvPath);

% filenames must stay row-aligned with the CSV for later error analysis, so
% they are written from the same trimmed, ordered list.
fid = fopen(namesPath, 'w');
if fid == -1
    error('feature_extractor:cannotWrite', 'Cannot open %s', namesPath);
end
cleanupFid = onCleanup(@() fclose(fid));
for k = 1:nProcessed
    fprintf(fid, '%s\n', processedPaths{k});
end
clear cleanupFid;

%% -------------------------------------------------------------- summary
fprintf('\n===================== SUMMARY =====================\n');
fprintf('Total images found      : %d\n', totalImages);
fprintf('Successfully processed  : %d\n', nProcessed);
fprintf('Skipped (errors)        : %d\n', nSkipped);
fprintf('Output matrix size      : %d x %d\n', size(featureMatrix,1), size(featureMatrix,2));
fprintf('Elapsed time            : %.1f s\n', elapsed);
fprintf('Feature CSV             : %s\n', csvPath);
fprintf('Filename list           : %s\n', namesPath);
fprintf('===================================================\n');
fprintf('\nNext: train_model.m to fit the classifier, then demo_image.m to\n');
fprintf('demonstrate it on a single image.\n');


%% ================================================================
%  Local functions
%  ================================================================

function files = listImages(folderPath)
%LISTIMAGES Full paths of the image files in a folder.
%   The dataset is nominally .jpg, but scraped folders are inconsistent and
%   dir() is case-sensitive on Linux/macOS, so match the extension with a
%   case-insensitive regexp instead of a glob.

    d = dir(folderPath);
    d = d(~[d.isdir]);

    if isempty(d)
        files = cell(0,1);
        return;
    end

    names   = {d.name}';
    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    names   = names(isImage);

    if isempty(names)
        files = cell(0,1);
        return;
    end

    files = fullfile(folderPath, names);
    files = sort(files);          % deterministic row order across runs
    files = files(:);
end


function [files, nDropped] = sampleAugmented(files, frac, seed)
%SAMPLEAUGMENTED Keep every original image; keep a random FRAC of the
%   augmented siblings make_augmented.m added (files whose name ends in
%   _qhi, _qlo, _rweb, or the legacy _q85/_q60/_r75q85).
%
%   Sampling is seeded, so re-running with the same FRAC picks the same
%   subset - the extracted CSV stays reproducible.

    [~, stems] = cellfun(@fileparts, files, 'UniformOutput', false);
    isAug = ~cellfun(@isempty, ...
        regexpi(stems, '_(qhi|qlo|rweb|q85|q60|r75q85)$', 'once'));

    originals = files(~isAug);
    augmented = files(isAug);

    rng(seed);
    nKeep = round(numel(augmented) * frac);
    nDropped = numel(augmented) - nKeep;
    if nDropped > 0
        keepIdx   = sort(randperm(numel(augmented), nKeep));
        augmented = augmented(keepIdx);
    end

    files = sort([originals; augmented]);
end
