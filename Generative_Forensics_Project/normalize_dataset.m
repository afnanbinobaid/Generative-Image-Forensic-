function normalize_dataset(outRoot, cropSide, targetSide, quality)
%NORMALIZE_DATASET  Strip the container confound before measuring anything.
%
%   normalize_dataset
%   normalize_dataset('Dataset_norm')
%   normalize_dataset('Dataset_norm', 448, 320, 85)
%
%   audit_encoding.py found this dataset separable at ROC-AUC 0.9969 by a
%   classifier that never reads a pixel - better than the 0.961 the DSP
%   detector reaches reading all 230 features. Three container properties are
%   perfectly correlated with the label:
%
%     chroma subsampling   real 4:4:4, AI 4:2:0 - 100% class-exclusive
%     bytes per pixel      0.669 against 0.367 - a 1.82x gap
%     native width         450-500 against 512-1024 - ranges do not overlap
%
%   None of that is about generation, and all of it reaches the features.
%   Halving chroma resolution and quantising harder both remove fine detail,
%   which is measured directly by the cD1 standard deviations in columns 95,
%   144 and 193 - the three most discriminative features in the whole set.
%
%   WHY THIS IS NOT SIMPLY "RE-ENCODE BOTH THE SAME WAY"
%
%   That was the obvious fix and it does not work. The damage is already in
%   the pixels: the AI class was written at a lower quality and lost fine
%   detail permanently, and no re-encode restores it. Worse, re-encoding an
%   image whose prior quantisation is still grid-aligned produces a
%   double-compression signature that depends on the PRIOR quality - so it
%   replaces one class-correlated artefact with a different one. Measured, on
%   images with identical content and only the encoding history differing:
%
%     treatment                         cD1 ratio   Cohen d
%     untouched                              2.76     19.47
%     crop + re-encode q85                   1.59      6.24   still separable
%     crop + re-encode q70                   0.84      2.46   inverted
%     crop + re-encode q60                   0.43     10.42   badly inverted
%
%   The fix is the one section 8 of the record already hinted at, in
%   explaining why web_sim flipped less often than jpeg85: RESAMPLING FIRST
%   DESTROYS THE 8x8 DCT GRID, and with it the prior quantisation signature.
%   Cropping to a common pixel size before resampling is what makes the
%   resample factor identical for both classes - which is precisely what
%   section 4 got wrong when it scaled two different-sized classes to one
%   size and manufactured a high-frequency difference.
%
%     crop448 -> resize 400 -> q85           1.14      1.95   marginal
%     crop448 -> resize 352 -> q85           1.05      0.74   closed
%     crop448 -> resize 320 -> q85           1.01      0.24   closed
%
%   A factor of about 1.25x is where it closes; the 1.4x default leaves
%   margin. Overall detail level is barely touched - the mean cD1 across both
%   classes stays within a few percent of untreated - so the cost is far
%   smaller than the confound it removes.
%
%   WHAT IT DOES, identically to every image in both classes:
%
%     1. centre-crop to cropSide at NATIVE SCALE, origin snapped to a
%        multiple of 8. Nothing is resampled yet; this only makes the two
%        classes the same pixel size so step 2 can use one factor for both.
%     2. resample to targetSide. One filter, one factor, both classes.
%     3. decimate chroma explicitly - RGB to YCbCr, box-downsample Cb and Cr
%        by two, bilinearly back up, to RGB. Stated in pixels rather than
%        left to imwrite, which picks subsampling from the quality value
%        internally and does not expose it.
%     4. write JPEG at one fixed quality with a .jpg extension for both
%        classes, so quantisation, bytes per pixel and the extension all stop
%        carrying label information.
%
%   NOTHING IS OVERWRITTEN. Output goes to a new tree, so the before/after
%   comparison stays possible and a mistake here costs nothing.
%
%   AFTER IT FINISHES
%
%     python audit_encoding.py Dataset_norm/Real_Images Dataset_norm/AI_Images
%
%   Expect ROC-AUC near 0.50 - that is the pass condition. Then:
%
%     rename Dataset Dataset_original
%     rename Dataset_norm Dataset
%     make_augmented                        (MATLAB)
%     feature_extractor                     (MATLAB)
%     python train_model.py
%
%   EXPECT THE ACCURACY TO FALL. That fall is the measurement: the share of
%   the old 89.4% that was provenance rather than generation. Report both.
%
%   TWO LIMITS THIS CANNOT FIX, both of which belong in the write-up:
%     - Field of view. A crop covers a different fraction of a 1024px frame
%       than of a 500px one, so the classes still see different amounts of
%       scene. Only a dataset with matched native resolutions fixes that.
%     - Signal above the new Nyquist is gone. The resample low-passes both
%       classes equally, so it adds no confound, but any generation artefact
%       living in the top 1.4x of the spectrum is no longer measurable.

    if nargin < 1 || isempty(outRoot),    outRoot    = 'Dataset_norm'; end
    if nargin < 2 || isempty(cropSide),   cropSide   = 448;            end
    if nargin < 3 || isempty(targetSide), targetSide = 320;            end
    if nargin < 4 || isempty(quality),    quality    = 85;             end

    if mod(cropSide, 8) ~= 0
        error('normalize_dataset:badCrop', ...
              'cropSide must be a multiple of 8 to keep DCT alignment (got %d).', cropSide);
    end
    if targetSide < 256
        error('normalize_dataset:tooSmall', ...
              ['targetSide must be at least 256 - the extractor measures a ' ...
               '256x256 window (got %d).'], targetSide);
    end
    if cropSide / targetSide < 1.25
        error('normalize_dataset:weakResample', ...
              ['cropSide/targetSide is %.2fx. Below about 1.25x the resample no ' ...
               'longer\ndestroys the prior DCT grid and the confound survives - ' ...
               'measured at\n1.02x (d=4.33), 1.12x (d=1.95), 1.27x (d=0.74), ' ...
               '1.40x (d=0.24).'], cropSide / targetSide);
    end

    classes = {'Real_Images', 'AI_Images'};
    srcRoot = 'Dataset';

    fprintf('\n');
    fprintf('======================================================================\n');
    fprintf('  CONTAINER NORMALISATION\n');
    fprintf('======================================================================\n');
    fprintf('  source      : %s\n', fullfile(pwd, srcRoot));
    fprintf('  destination : %s   (created; nothing is overwritten)\n', fullfile(pwd, outRoot));
    fprintf('  crop        : %d x %d, native scale, no resampling\n', cropSide, cropSide);
    fprintf('  resample    : -> %d (%.2fx), one factor for both classes\n', ...
            targetSide, cropSide / targetSide);
    fprintf('  chroma      : decimated 2x explicitly, both classes\n');
    fprintf('  encode      : JPEG quality %d, .jpg, both classes\n\n', quality);

    grand = struct('written', 0, 'skippedSmall', 0, 'skippedAug', 0, ...
                   'existed', 0, 'failed', 0);

    for k = 1:numel(classes)
        srcDir = fullfile(pwd, srcRoot, classes{k});
        dstDir = fullfile(pwd, outRoot, classes{k});

        if ~isfolder(srcDir)
            error('normalize_dataset:noSource', 'Cannot find %s', srcDir);
        end
        if ~isfolder(dstDir)
            mkdir(dstDir);
        end

        d = dir(srcDir);
        d = d(~[d.isdir]);
        names = {d.name}';

        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        % Augmented copies are regenerated from the normalised originals;
        % carrying the old ones across would stack a second compression on
        % images that have already been through one.
        isMade  = ~cellfun(@isempty, ...
                           regexpi(names, '_(qhi|qlo|rweb|q85|q60|r75q85)\.', 'once'));
        skippedAug = sum(isImage & isMade);
        names = sort(names(isImage & ~isMade));

        fprintf('  %s: %d originals (%d augmented copies left behind)\n', ...
                classes{k}, numel(names), skippedAug);

        written = 0; skippedSmall = 0; existed = 0; failed = 0;
        seen = containers.Map('KeyType', 'char', 'ValueType', 'char');

        for i = 1:numel(names)
            src = fullfile(srcDir, names{i});
            [~, base] = fileparts(names{i});
            dst = fullfile(dstDir, [base '.jpg']);

            % Two originals differing only by extension would collapse onto
            % one output name and silently lose the second - the same hazard
            % make_augmented.m carries. Renamed here rather than dropped.
            if isKey(seen, lower(base))
                warning('normalize_dataset:collision', ...
                        '%s and %s share a stem; writing the second as %s__2.jpg', ...
                        seen(lower(base)), names{i}, base);
                dst = fullfile(dstDir, [base '__2.jpg']);
            else
                seen(lower(base)) = names{i};
            end

            if isfile(dst)
                existed = existed + 1;
                continue;
            end

            try
                img  = toUint8Rgb(imread(src));
                crop = centreCrop(img, cropSide);
                if isempty(crop)
                    skippedSmall = skippedSmall + 1;
                    continue;
                end

                small = imresize(crop, [targetSide targetSide], 'bicubic');
                imwrite(decimateChroma(small), dst, 'Quality', quality);
                written = written + 1;

            catch err
                failed = failed + 1;
                fprintf(2, '    %s: %s\n', names{i}, err.message);
            end

            if mod(i, 500) == 0
                fprintf('    %d / %d\n', i, numel(names));
            end
        end

        fprintf('    written %d   already there %d   too small %d   failed %d\n\n', ...
                written, existed, skippedSmall, failed);

        grand.written      = grand.written + written;
        grand.existed      = grand.existed + existed;
        grand.skippedSmall = grand.skippedSmall + skippedSmall;
        grand.skippedAug   = grand.skippedAug + skippedAug;
        grand.failed       = grand.failed + failed;
    end

    fprintf('----------------------------------------------------------------------\n');
    fprintf('  written %d   already there %d   too small %d   failed %d\n', ...
            grand.written, grand.existed, grand.skippedSmall, grand.failed);

    if grand.skippedSmall > 0
        fprintf(2, ['\n  %d image(s) were smaller than %dpx and were DROPPED rather than\n' ...
                    '  upscaled. Upscaling is a low-pass filter and would reintroduce the\n' ...
                    '  confound this script exists to remove. If the drops fall mostly in\n' ...
                    '  one class, say so when reporting - it is a selection effect.\n'], ...
                grand.skippedSmall, cropSide);
    end

    fprintf('\n  Next:\n');
    fprintf('    python audit_encoding.py %s/Real_Images %s/AI_Images\n', outRoot, outRoot);
    fprintf('    -> expect ROC-AUC near 0.50. Above ~0.65 means a container cue\n');
    fprintf('       survived and must be found before retraining.\n\n');
end


%% ================================================================
%  Local functions
%  ================================================================

function img = toUint8Rgb(img)
%TOUINT8RGB  Force any imread output into 3-channel uint8 RGB.
%   Matches extractImageFeatures.m, so the normalised set is built on the same
%   convention the features are measured with.

    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end

    nCh = size(img, 3);
    if nCh == 1
        img = repmat(img, [1 1 3]);          % grayscale -> RGB
    elseif nCh == 2
        img = repmat(img(:,:,1), [1 1 3]);   % gray + alpha -> drop alpha
    elseif nCh >= 4
        img = img(:,:,1:3);                  % RGBA / CMYK-ish -> drop extras
    end
end


function out = centreCrop(img, side)
%CENTRECROP  A side x side centre crop at native scale, or [] if too small.
%   The origin snaps to a multiple of 8 so the crop stays aligned with the
%   JPEG DCT block grid, exactly as extractImageFeatures.m does.

    [h, w, ~] = size(img);
    if h < side || w < side
        out = [];
        return;
    end

    r0 = floor((h - side) / 2);
    c0 = floor((w - side) / 2);
    r0 = r0 - mod(r0, 8) + 1;
    c0 = c0 - mod(c0, 8) + 1;

    out = img(r0:r0+side-1, c0:c0+side-1, :);
end


function img = decimateChroma(img)
%DECIMATECHROMA  Halve the chroma resolution, the way 4:2:0 does.
%
%   Done in pixels rather than by asking the encoder for 4:2:0, because
%   MATLAB's imwrite picks subsampling from the quality value internally and
%   does not expose it. Stating the operation here makes it reproducible and
%   inspectable instead of a side effect of an encoder setting.
%
%   Applied to an image that was already 4:2:0 it changes very little - the
%   detail it removes is already gone - which is what makes it safe to run on
%   both classes rather than only on the one that needs it.

    ycc = rgb2ycbcr(img);
    y   = ycc(:,:,1);
    cb  = ycc(:,:,2);
    cr  = ycc(:,:,3);

    full = size(cb);
    cb = imresize(imresize(cb, 0.5, 'box'), full, 'bilinear');
    cr = imresize(imresize(cr, 0.5, 'box'), full, 'bilinear');

    img = ycbcr2rgb(cat(3, y, cb, cr));
end
