function [features, crop, grayD] = extractImageFeatures(source, preprocess)
%EXTRACTIMAGEFEATURES  The 230-feature DSP vector for a single image.
%
%   features = extractImageFeatures(PATH) reads an image file and returns a
%   1x230 row vector in the documented block order:
%
%     1 - 51    Block A  spatial stats, per channel R,G,B (17 x 3)
%    52 - 198   Block B  2-level db4 DWT subband stats, per channel (49 x 3)
%   199 - 218   Block C  FFT radial power spectrum, luminance (20)
%   219 - 227   Block D  high-pass noise residual stats, luminance (9)
%   228 - 230   Block E  cross-channel correlation R-G, R-B, G-B (3)
%
%   features = extractImageFeatures(IMG) accepts an image array instead.
%   features = extractImageFeatures(..., PREPROCESS) selects 'crop' (default)
%   or 'resize'.
%
%   [features, crop, grayD] = ... also returns the 256x256 analysed region and
%   its luminance, so a caller can display exactly what was measured without
%   repeating the work.
%
%   This function is the single definition of the feature set. feature_extractor.m
%   (batch) and demo_image.m (live) both call it, so the numbers a demo shows can
%   never drift away from the numbers the model was trained on.
%
%   Requires: Image Processing, Wavelet, and Statistics & Machine Learning
%   Toolboxes.

    if nargin < 2 || isempty(preprocess)
        preprocess = 'crop';
    end

    IMG_SIZE    = [256 256];
    NUM_RINGS   = 20;
    GLCM_LEVELS = 8;
    WAVELET     = 'db4';
    WAVE_LEVEL  = 2;
    NUM_FEATURES = 230;

    %% --- read ---------------------------------------------------------
    if ischar(source) || isstring(source)
        % imread with a colormap output so indexed images are handled too.
        [img, map] = imread(char(source));
        if ~isempty(map)
            img = ind2rgb(img, map);        % indexed -> RGB double [0 1]
        end
    else
        img = source;
    end

    img  = toUint8Rgb(img);                 % force 3-channel uint8 RGB
    crop = standardiseSize(img, IMG_SIZE, preprocess);   % no re-encode

    R = crop(:,:,1);
    G = crop(:,:,2);
    B = crop(:,:,3);

    % Luminance computed once, reused by Blocks C and D.
    grayD = double(rgb2gray(crop));

    %% --- the five blocks ----------------------------------------------
    blockA = [ spatialFeatures(R, GLCM_LEVELS), ...
               spatialFeatures(G, GLCM_LEVELS), ...
               spatialFeatures(B, GLCM_LEVELS) ];

    blockB = [ waveletFeatures(R, WAVELET, WAVE_LEVEL), ...
               waveletFeatures(G, WAVELET, WAVE_LEVEL), ...
               waveletFeatures(B, WAVELET, WAVE_LEVEL) ];

    blockC = radialSpectrum(grayD, NUM_RINGS);
    blockD = residualFeatures(grayD);
    blockE = channelCorrelation(R, G, B);

    features = [blockA, blockB, blockC, blockD, blockE];

    if numel(features) ~= NUM_FEATURES
        error('extractImageFeatures:badLength', ...
              'Expected %d features, got %d', NUM_FEATURES, numel(features));
    end

    % Backstop: degenerate cases are handled where they arise, so this should
    % never fire, but a non-finite value must never reach the CSV or the model.
    features(~isfinite(features)) = 0;
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


function img = standardiseSize(img, targetSize, mode)
%STANDARDISESIZE Bring an image to targetSize by cropping or by scaling.
%   'crop'   takes a centre crop, so pixels keep their native scale and
%            nothing is resampled. Use this whenever the two classes have
%            different native resolutions: scaling them by different factors
%            low-pass filters them by different amounts and manufactures a
%            high-frequency difference that has nothing to do with whether
%            an image was generated. Cropping also leaves the JPEG
%            compression history intact, which resizing partly destroys.
%   'resize' scales the whole frame. Keeps the full composition, but
%            resamples every pixel.

    [h, w, ~] = size(img);

    switch lower(mode)
        case 'resize'
            img = imresize(img, targetSize);

        case 'crop'
            % Scale up only when the image is too small to crop from.
            if h < targetSize(1) || w < targetSize(2)
                scale = max(targetSize(1)/h, targetSize(2)/w);
                img   = imresize(img, ceil([h w] * scale));
                [h, w, ~] = size(img);
            end

            % Snap the origin to a multiple of 8 so the crop stays aligned
            % with the JPEG DCT block grid.
            r0 = floor((h - targetSize(1)) / 2);
            c0 = floor((w - targetSize(2)) / 2);
            r0 = r0 - mod(r0, 8) + 1;
            c0 = c0 - mod(c0, 8) + 1;

            img = img(r0:r0+targetSize(1)-1, c0:c0+targetSize(2)-1, :);

        otherwise
            error('feature_extractor:badMode', ...
                  'PREPROCESS must be ''crop'' or ''resize'', got ''%s''', mode);
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

    % Argument order is (C, S, wname, N) - the coefficient vector and the
    % bookkeeping matrix come first, matching detcoef2's (O, C, S, N).
    cA2 = appcoef2(C, S, waveName, level);
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
