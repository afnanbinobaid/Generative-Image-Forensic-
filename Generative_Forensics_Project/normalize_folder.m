function stats = normalize_folder(srcDir, dstDir, cropSide, targetSide, quality)
%NORMALIZE_FOLDER  Put one folder of images through the container normalisation.
%
%   normalize_folder('E:\test\real', 'E:\test\real_norm')
%   normalize_folder(src, dst, 448, 320, 85)
%   stats = normalize_folder(...)
%
%   This is the worker. normalize_dataset.m calls it twice, once per class, and
%   it is also the way to prepare a TEST folder - which has to go through the
%   identical treatment the training set did, or the model is being shown a
%   kind of image it never learned.
%
%   Why the treatment is what it is, in short: audit_encoding.py found the
%   training set separable at ROC-AUC 0.9969 by a classifier that reads no
%   pixels, because the two classes carried different chroma subsampling,
%   different bytes per pixel and non-overlapping native widths. Re-encoding
%   alone does not fix that - it replaces one class-correlated artefact with a
%   different one, and at some qualities inverts the gap. What works is
%   cropping both to a common pixel size FIRST, so one resample factor serves
%   everything, then resampling: that destroys the 8x8 DCT grid and the prior
%   quantisation signature with it. See normalize_dataset.m for the measured
%   sweep behind those numbers.
%
%   THE SIZE FLOOR IS NOT A TUNING KNOB
%
%   Images smaller than cropSide are DROPPED, not upscaled - upscaling is a
%   low-pass filter and would reintroduce exactly the confound this removes.
%   And cropSide must be the SAME value the training set was built with. Lower
%   it for a test folder and those images have been through a different
%   pipeline, so any score from them is measuring the pipeline difference.
%   The floor is the model's operating range: below it, it has no answer.
%
%   Returns a struct: written, existed, skippedSmall, skippedAug, failed, and
%   droppedSizes (an N x 2 list of the [width height] that were too small).

    if nargin < 3 || isempty(cropSide),   cropSide   = 448; end
    if nargin < 4 || isempty(targetSide), targetSide = 320; end
    if nargin < 5 || isempty(quality),    quality    = 85;  end

    if ~isfolder(srcDir)
        error('normalize_folder:noSource', 'Cannot find %s', srcDir);
    end
    if mod(cropSide, 8) ~= 0
        error('normalize_folder:badCrop', ...
              'cropSide must be a multiple of 8 to keep DCT alignment (got %d).', cropSide);
    end
    if targetSide < 256
        error('normalize_folder:tooSmall', ...
              ['targetSide must be at least 256 - the extractor measures a ' ...
               '256x256 window (got %d).'], targetSide);
    end
    if cropSide / targetSide < 1.25
        error('normalize_folder:weakResample', ...
              ['cropSide/targetSide is %.2fx. Below about 1.25x the resample no ' ...
               'longer\ndestroys the prior DCT grid and the confound survives - ' ...
               'measured at\n1.02x (d=4.33), 1.12x (d=1.95), 1.27x (d=0.74), ' ...
               '1.40x (d=0.24).'], cropSide / targetSide);
    end

    if ~isfolder(dstDir)
        mkdir(dstDir);
    end

    d = dir(srcDir);
    d = d(~[d.isdir]);
    names = {d.name}';

    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    % Augmented copies are regenerated from the normalised originals; carrying
    % the old ones across would stack a second compression on images that have
    % already been through one.
    isMade  = ~cellfun(@isempty, ...
                       regexpi(names, '_(qhi|qlo|rweb|soft|q85|q60|r75q85)\.', 'once'));
    skippedAug = sum(isImage & isMade);
    names = sort(names(isImage & ~isMade));

    stats = struct('written', 0, 'existed', 0, 'skippedSmall', 0, ...
                   'skippedAug', skippedAug, 'failed', 0, 'droppedSizes', []);

    if isempty(names)
        fprintf('    %s: no images found\n', srcDir);
        return;
    end

    seen = containers.Map('KeyType', 'char', 'ValueType', 'char');
    dropped = zeros(0, 2);

    for i = 1:numel(names)
        src = fullfile(srcDir, names{i});
        [~, base] = fileparts(names{i});
        dst = fullfile(dstDir, [base '.jpg']);

        % Two originals differing only by extension would collapse onto one
        % output name and silently lose the second - the same hazard
        % make_augmented.m carries. Renamed here rather than dropped.
        if isKey(seen, lower(base))
            warning('normalize_folder:collision', ...
                    '%s and %s share a stem; writing the second as %s__2.jpg', ...
                    seen(lower(base)), names{i}, base);
            dst = fullfile(dstDir, [base '__2.jpg']);
        else
            seen(lower(base)) = names{i};
        end

        if isfile(dst)
            stats.existed = stats.existed + 1;
            continue;
        end

        try
            % normalize_image.m owns the treatment; demo_image.m calls the same
            % function, so a live upload and a training image cannot drift apart.
            if normalize_image(src, dst, cropSide, targetSide, quality)
                stats.written = stats.written + 1;
            else
                info = imfinfo(src);
                stats.skippedSmall = stats.skippedSmall + 1;
                dropped(end+1, :) = [info(1).Width info(1).Height];  %#ok<AGROW>
            end

        catch err
            stats.failed = stats.failed + 1;
            fprintf(2, '    %s: %s\n', names{i}, err.message);
        end

        if mod(i, 500) == 0
            fprintf('    %d / %d\n', i, numel(names));
        end
    end

    stats.droppedSizes = dropped;

    %% ---------------------------------------------------------- summary
    fprintf('  %s\n', srcDir);
    fprintf('    written %d   already there %d   too small %d   failed %d\n', ...
            stats.written, stats.existed, stats.skippedSmall, stats.failed);

    kept = stats.written + stats.existed;
    seen = kept + stats.skippedSmall + stats.failed;
    if seen > 0
        fprintf('    kept %d of %d (%.0f%%)\n', kept, seen, 100 * kept / seen);
    end

    % A dropped image is not a rounding error - it is a photograph the model
    % has no answer for. On a test folder the drops decide what the accuracy
    % afterwards is even measuring, so their size is reported rather than
    % left as a count.
    if stats.skippedSmall > 0
        shortSide = min(dropped, [], 2);
        fprintf(2, ['    %d image(s) were under %dpx on the short side and were DROPPED.\n' ...
                    '      their short sides: min %d, median %d, max %d\n' ...
                    '      Dropped, not upscaled - upscaling is a low-pass filter and would\n' ...
                    '      push them toward the AI verdict for a reason unrelated to origin.\n'], ...
                stats.skippedSmall, cropSide, ...
                min(shortSide), round(median(shortSide)), max(shortSide));

        if kept > 0 && stats.skippedSmall > kept
            fprintf(2, ['      MORE WERE DROPPED THAN KEPT. Whatever is measured on what\n' ...
                        '      survives is a statement about large images only, not about\n' ...
                        '      this folder.\n']);
        end
    end
end
