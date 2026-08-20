%% feature_extractor.m
% Classical DSP feature extraction for AI-generated image forensics.
%
% Walks Dataset/Real_Images (label 0) and Dataset/AI_Images (label 1),
% extracts a fixed 1x230 feature vector per image, appends the label, and
% writes the result to dataset.csv (no header, pure numeric).
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
IMG_SIZE      = [256 256];   % all images resized to this
NUM_FEATURES  = 230;         % feature count, excluding the label
NUM_RINGS     = 20;          % Block C radial bins
GLCM_LEVELS   = 8;           % graycomatrix quantisation levels
WAVELET       = 'db4';
WAVE_LEVEL    = 2;
PROGRESS_STEP = 100;         % print a progress line every N images

projectRoot = pwd;
realDir     = fullfile(projectRoot, 'Dataset', 'Real_Images');
aiDir       = fullfile(projectRoot, 'Dataset', 'AI_Images');
csvPath     = fullfile(projectRoot, 'dataset.csv');
namesPath   = fullfile(projectRoot, 'filenames.txt');

if ~isfolder(realDir)
    error('feature_extractor:missingFolder', 'Cannot find %s', realDir);
end
if ~isfolder(aiDir)
    error('feature_extractor:missingFolder', 'Cannot find %s', aiDir);
end

%% ------------------------------------------------------------- file list
realFiles = listImages(realDir);
aiFiles   = listImages(aiDir);

% Concatenate both classes up front so progress reporting reflects the
% combined total rather than restarting per folder.
allFiles  = [realFiles;               aiFiles];
allLabels = [zeros(numel(realFiles),1); ones(numel(aiFiles),1)];

totalImages = numel(allFiles);
if totalImages == 0
    error('feature_extractor:noImages', ...
          'No images found under %s or %s', realDir, aiDir);
end

fprintf('Found %d images (%d real, %d AI).\n', ...
        totalImages, numel(realFiles), numel(aiFiles));
fprintf('Extracting %d features per image...\n\n', NUM_FEATURES);

%% ------------------------------------------------------------ main loop
% Pre-allocate the full matrix; trim to the processed rows at the end.
featureMatrix = NaN(totalImages, NUM_FEATURES + 1);
processedPaths = cell(totalImages, 1);

nProcessed  = 0;
nSkipped    = 0;
nSanitized  = 0;   % non-finite feature values that had to be replaced (expected: 0)

startTime = tic;

for k = 1:totalImages
    imgPath = allFiles{k};

    try
        %% --- preprocessing -------------------------------------------
        % imread with a colormap output so indexed images are handled too.
        [img, map] = imread(imgPath);
        if ~isempty(map)
            img = ind2rgb(img, map);        % indexed -> RGB double [0 1]
        end

        img = toUint8Rgb(img);              % force 3-channel uint8 RGB
        img = imresize(img, IMG_SIZE);      % no re-encode, no grayscale

        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);

        % Luminance computed once, reused by Blocks C and D.
        grayD = double(rgb2gray(img));

        %% --- Block A: global spatial features (51) --------------------
        blockA = [ spatialFeatures(R, GLCM_LEVELS), ...
                   spatialFeatures(G, GLCM_LEVELS), ...
                   spatialFeatures(B, GLCM_LEVELS) ];

        %% --- Block B: 2-level DWT subband features (147) --------------
        blockB = [ waveletFeatures(R, WAVELET, WAVE_LEVEL), ...
                   waveletFeatures(G, WAVELET, WAVE_LEVEL), ...
                   waveletFeatures(B, WAVELET, WAVE_LEVEL) ];

        %% --- Block C: FFT radial power spectrum (20) ------------------
        blockC = radialSpectrum(grayD, NUM_RINGS);

        %% --- Block D: high-pass noise residual stats (9) --------------
        blockD = residualFeatures(grayD);

        %% --- Block E: cross-channel correlation (3) -------------------
        blockE = channelCorrelation(R, G, B);

        %% --- assemble ------------------------------------------------
        featureVector = [blockA, blockB, blockC, blockD, blockE];

        if numel(featureVector) ~= NUM_FEATURES
            error('feature_extractor:badLength', ...
                  'Expected %d features, got %d', ...
                  NUM_FEATURES, numel(featureVector));
        end

        % Backstop only: the degenerate cases (flat channels, all-zero
        % subbands) are already handled where they arise, so this should
        % stay at zero. Keep the count so the summary says otherwise if a
        % non-finite value ever slips through.
        bad = ~isfinite(featureVector);
        if any(bad)
            nSanitized = nSanitized + sum(bad);
            featureVector(bad) = 0;
        end

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

% filenames.txt must stay row-aligned with dataset.csv for later error
% analysis, so it is written from the same trimmed, ordered list.
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
if nSanitized > 0
    fprintf('Non-finite values fixed : %d (replaced with 0)\n', nSanitized);
end
fprintf('Elapsed time            : %.1f s\n', elapsed);
fprintf('Feature CSV             : %s\n', csvPath);
fprintf('Filename list           : %s\n', namesPath);
fprintf('===================================================\n');


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


function img = toUint8Rgb(img)
%TOUINT8RGB Force any imread output into a 3-channel uint8 RGB image.

    % Class normalisation first: 16-bit PNGs, logical masks and the double
    % output of ind2rgb all become uint8 here.
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end

    nCh = size(img, 3);
    if nCh == 1
        img = repmat(img, [1 1 3]);      % grayscale -> RGB
    elseif nCh == 2
        img = repmat(img(:,:,1), [1 1 3]);   % gray+alpha -> drop alpha
    elseif nCh >= 4
        img = img(:,:,1:3);              % RGBA (or CMYK-ish) -> drop extras
    end
end


function stats = computeBaseStats(data)
%COMPUTEBASESTATS Seven first/second-order statistics of any 2D matrix.
%   stats = [Mean, Std, Variance, Energy, Entropy, Skewness, Kurtosis]
%
%   Works on both uint8 image planes and signed, non-integer wavelet
%   coefficients, which is why entropy is estimated from a histogram rather
%   than with the built-in entropy().

    x = double(data(:));

    mu  = mean(x);
    sd  = std(x);
    vr  = var(x);
    en  = sum(x.^2) / numel(x);
    ent = shannonEntropy(x, 256);   % fixed 256-bin histogram estimator

    % Skewness and kurtosis are 0/0 for constant data - which a wavelet
    % detail subband of a flat image genuinely is. Test the deviations
    % rather than sd: on constant data with a large mean, cancellation can
    % leave std() reporting a spurious ~1e-26 instead of an exact zero, so
    % an "sd == 0" check would miss the degenerate case and let a NaN through.
    [sk, ku] = higherMoments(x, mu);

    stats = [mu, sd, vr, en, ent, sk, ku];
end


function [sk, ku] = higherMoments(x, mu)
%HIGHERMOMENTS Skewness and kurtosis, with the degenerate cases zeroed.

    if ~any(x - mu)             % every sample identical -> moments undefined
        sk = 0;
        ku = 0;
        return;
    end

    sk = skewness(x);
    ku = kurtosis(x);

    % Backstop for anything still pathological (e.g. deviations so small the
    % normalising sd^3 / sd^4 underflows).
    if ~isfinite(sk), sk = 0; end
    if ~isfinite(ku), ku = 0; end
end


function H = shannonEntropy(x, nBins)
%SHANNONENTROPY Histogram-based Shannon entropy, in bits.
%   Bins the data over its own range and evaluates -sum(p .* log2(p)) over
%   the non-empty bins only. Valid for arbitrary real-valued input.

    x = double(x(:));
    x = x(isfinite(x));

    if isempty(x)
        H = 0;
        return;
    end

    lo = min(x);
    hi = max(x);
    if lo == hi                 % constant data carries no information
        H = 0;
        return;
    end

    counts = histcounts(x, linspace(lo, hi, nBins + 1));
    p      = counts(counts > 0) / sum(counts);
    H      = -sum(p .* log2(p));
end


function feats = spatialFeatures(channel, glcmLevels)
%SPATIALFEATURES Block A features for one colour channel (1x17).
%   [7 base stats, 4 GLCM stats, 6 Sobel gradient stats]

    chD = double(channel);

    % --- 7 base stats
    base = computeBaseStats(chD);

    % --- 4 GLCM texture stats
    % Fixed GrayLimits so the quantisation is identical across images
    % instead of stretching to each image's own min/max.
    glcm  = graycomatrix(channel, ...
                         'NumLevels', glcmLevels, ...
                         'GrayLimits', [0 255], ...
                         'Offset', [0 1], ...
                         'Symmetric', true);
    props = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});

    % Correlation is undefined for a constant channel (zero GLCM variance).
    corrVal = props.Correlation;
    if ~isfinite(corrVal)
        corrVal = 0;
    end

    glcmStats = [props.Contrast, corrVal, props.Energy, props.Homogeneity];

    % --- 6 Sobel edge stats
    [Gx, Gy] = imgradientxy(chD, 'sobel');
    Gmag     = imgradient(Gx, Gy);

    edgeStats = [mean(Gx(:)), var(Gx(:)), ...
                 mean(Gy(:)), var(Gy(:)), ...
                 mean(Gmag(:)), var(Gmag(:))];

    feats = [base, glcmStats, edgeStats];
end


function feats = waveletFeatures(channel, waveName, level)
%WAVELETFEATURES Block B features for one colour channel (1x49).
%   Base stats for cA2, cH2, cV2, cD2, cH1, cV1, cD1 in that order.

    chD = double(channel);

    [C, S] = wavedec2(chD, level, waveName);

    cA2 = appcoef2(level, C, S, waveName);
    [cH2, cV2, cD2] = detcoef2('all', C, S, 2);
    [cH1, cV1, cD1] = detcoef2('all', C, S, 1);

    subbands = {cA2, cH2, cV2, cD2, cH1, cV1, cD1};

    feats = zeros(1, 7 * numel(subbands));
    for i = 1:numel(subbands)
        idx = (i-1)*7 + (1:7);
        feats(idx) = computeBaseStats(subbands{i});
    end
end


function feats = radialSpectrum(grayD, nRings)
%RADIALSPECTRUM Block C: mean FFT magnitude in nRings concentric rings.
%   Ring 1 is centred on DC (low frequency); ring nRings is the outermost
%   (high frequency) band.

    magSpec = abs(fftshift(fft2(grayD)));

    [rows, cols] = size(magSpec);

    % fftshift puts the zero-frequency component at these indices.
    cy = floor(rows/2) + 1;
    cx = floor(cols/2) + 1;

    [X, Y] = meshgrid(1:cols, 1:rows);
    radius = sqrt((X - cx).^2 + (Y - cy).^2);

    rMax = max(radius(:));

    % Map every pixel into [1, nRings]; the +eps guard keeps the single
    % outermost pixel from landing in a non-existent ring nRings+1.
    ringIdx = floor(radius / (rMax + eps(rMax)) * nRings) + 1;
    ringIdx = min(max(ringIdx, 1), nRings);

    ringMeans = accumarray(ringIdx(:), magSpec(:), [nRings 1], @mean, 0);

    feats = ringMeans(:)';
end


function feats = residualFeatures(grayD)
%RESIDUALFEATURES Block D: mean/std/kurtosis of 3 high-pass residuals (1x9).

    % 1) 3x3 Laplacian high-pass
    lapKernel = [ 0 -1  0
                 -1  4 -1
                  0 -1  0];
    resLap = imfilter(grayD, lapKernel, 'replicate', 'same', 'conv');

    % 2) median residual
    resMed = grayD - medfilt2(grayD, [3 3], 'symmetric');

    % 3) Gaussian-blur high-pass residual
    resGau = grayD - imgaussfilt(grayD, 1);

    feats = [residualStats(resLap), residualStats(resMed), residualStats(resGau)];
end


function s = residualStats(r)
%RESIDUALSTATS Mean, standard deviation and kurtosis of a residual image.

    x  = double(r(:));
    mu = mean(x);
    sd = std(x);

    [~, ku] = higherMoments(x, mu);   % 0 for a perfectly flat residual

    s = [mu, sd, ku];
end


function feats = channelCorrelation(R, G, B)
%CHANNELCORRELATION Block E: Pearson r for R-G, R-B and G-B (1x3).

    r = double(R(:));
    g = double(G(:));
    b = double(B(:));

    feats = [pearson(r, g), pearson(r, b), pearson(g, b)];
end


function rho = pearson(a, b)
%PEARSON Pearson correlation coefficient, 0 when either input is constant.

    if std(a) == 0 || std(b) == 0
        rho = 0;
        return;
    end

    c   = corrcoef(a, b);
    rho = c(1,2);

    if ~isfinite(rho)
        rho = 0;
    end
end
