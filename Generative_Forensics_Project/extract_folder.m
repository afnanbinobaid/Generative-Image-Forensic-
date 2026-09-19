function extract_folder(folderPath, outCsv, mode)
%EXTRACT_FOLDER  Normalise a folder of images, measure them, write a feature CSV.
%
%   extract_folder('E:\test\ai', 'ai.csv')
%   extract_folder('E:\test\real', 'real.csv')
%   extract_folder(folder, outCsv, 'raw')        % skip normalisation
%
%   Then score it in Python, saying what the images actually are:
%
%       python score_folder.py ai.csv ai
%       python score_folder.py real.csv real
%       python score_folder.py ai.csv              (predictions only)
%
%   WHY THIS NORMALISES BEFORE MEASURING
%
%   The model was trained on images put through normalize_image.m - a 448px
%   centre crop, resampled to 320, chroma decimated, re-encoded at quality 85.
%   Measuring a raw folder and scoring it against that model is asking the
%   model about a kind of picture it has never seen: a 2000px photograph
%   cropped at native scale carries far more fine detail than a 320px training
%   image does, and this detector reads fine detail. The number that comes back
%   is real-looking and means nothing. That defect sat in the live demo for
%   weeks without producing a single error message.
%
%   So this function calls normalize_folder.m, which calls normalize_image.m -
%   the same one definition of the treatment the training set and the demo both
%   use. The normalised copies are written to <folderPath>_norm and kept, so a
%   second run is nearly free and you can look at what was actually measured.
%
%   THE THIRD ARGUMENT
%
%     'auto'      (default) normalise, unless the folder already looks
%                 normalised - every image exactly 320x320 - in which case
%                 measure it as it is.
%     'normalise' always normalise. Use if auto guessed wrong.
%     'raw'       never normalise. Only for a folder you normalised yourself,
%                 or for deliberately measuring the un-normalised images to
%                 compare. Any score from raw images is not comparable to the
%                 model's reported accuracy.
%
%   IMAGES BELOW THE FLOOR ARE DROPPED, NOT UPSCALED
%
%   Anything under 448px on the short side cannot be given the training set's
%   treatment, so normalize_folder drops it and this reports how many. That
%   matters for reading the result: if a third of the folder was dropped, the
%   accuracy afterwards is a statement about the large images only. Upscaling
%   instead would be a low-pass filter and would push small images toward the
%   AI verdict for a reason that has nothing to do with their origin.

    if nargin < 2 || isempty(outCsv)
        outCsv = 'folder_features.csv';
    end
    if nargin < 3 || isempty(mode)
        mode = 'auto';
    end
    mode = lower(mode);
    if ~ismember(mode, {'auto', 'normalise', 'normalize', 'raw'})
        error('extract_folder:badMode', ...
              'mode must be ''auto'', ''normalise'' or ''raw'' (got ''%s'').', mode);
    end
    if ~isfolder(folderPath)
        error('extract_folder:noFolder', 'Not a folder: %s', folderPath);
    end

    %% ------------------------------------------------ normalise, or not
    measureDir = folderPath;
    nDropped   = 0;

    if strcmp(mode, 'auto')
        if looksNormalised(folderPath)
            fprintf(['This folder already looks normalised (every image 320x320).\n' ...
                     'Measuring it as it is. Pass ''normalise'' to force.\n\n']);
            mode = 'raw';
        else
            mode = 'normalise';
        end
    end

    if ~strcmp(mode, 'raw')
        normDir = [regexprep(folderPath, '[\\/]+$', '') '_norm'];
        fprintf('Normalising into %s\n', normDir);
        fprintf('  (448 crop -> 320 resample -> chroma decimate -> JPEG 85)\n\n');

        stats = normalize_folder(folderPath, normDir);

        measureDir = normDir;
        nDropped   = stats.skippedSmall + stats.failed;

        if stats.written + stats.existed == 0
            error('extract_folder:allDropped', ...
                  ['Every image in %s was dropped or failed.\n' ...
                   'Nothing can be measured. If these images are all under 448px ' ...
                   'on the short\nside, they are outside the model''s operating ' ...
                   'range and there is no verdict for them.'], folderPath);
        end
        fprintf('\n');
    end

    %% ------------------------------------------------ measure
    d = dir(measureDir);
    d = d(~[d.isdir]);
    names = {d.name}';
    if ~isempty(names)
        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        names = sort(names(isImage));
    end
    if isempty(names)
        error('extract_folder:noImages', ...
              'No .jpg/.jpeg/.png images in %s', measureDir);
    end

    fprintf('Measuring %d images from %s\n', numel(names), measureDir);

    matrix = NaN(numel(names), 230);
    kept   = cell(numel(names), 1);
    nKept  = 0;
    nSkip  = 0;

    for i = 1:numel(names)
        f = fullfile(measureDir, names{i});
        try
            nKept = nKept + 1;
            matrix(nKept, :) = extractImageFeatures(f, 'crop');
            kept{nKept} = names{i};
        catch ME
            nKept = nKept - 1;
            nSkip = nSkip + 1;
            fprintf(2, 'SKIPPED: %s  (%s)\n', names{i}, ME.message);
        end
        if mod(i, 50) == 0
            fprintf('  %d / %d\n', i, numel(names));
        end
    end

    matrix = matrix(1:nKept, :);
    kept   = kept(1:nKept);

    if nKept == 0
        error('extract_folder:nothingMeasured', ...
              'Every image in %s failed to measure.', measureDir);
    end

    writematrix(matrix, outCsv);

    [p, base] = fileparts(outCsv);
    namesPath = fullfile(p, [base '.filenames.txt']);
    fid = fopen(namesPath, 'w');
    if fid ~= -1
        cleanup = onCleanup(@() fclose(fid));
        for i = 1:nKept
            fprintf(fid, '%s\n', kept{i});
        end
        clear cleanup;
    end

    %% ------------------------------------------------ summary
    fprintf('\nMeasured %d, failed %d\n', nKept, nSkip);
    fprintf('Wrote %s (%d x %d)\n', fullfile(pwd, outCsv), nKept, 230);

    % A drop is not a rounding error. It decides what the score is about.
    if nDropped > 0
        total = nKept + nDropped;
        fprintf(2, ['\n%d of %d images (%.0f%%) were dropped before measuring - ' ...
                    'under 448px\non the short side, outside the model''s operating ' ...
                    'range.\nWhatever you score next is a statement about the %d ' ...
                    'that survived.\n'], ...
                nDropped, total, 100 * nDropped / total, nKept);
    end

    fprintf('\nNow score it - say what these images actually are:\n');
    fprintf('    python score_folder.py %s ai\n', outCsv);
    fprintf('    python score_folder.py %s real\n', outCsv);
end


%% ================================================================
%  Local functions
%  ================================================================

function tf = looksNormalised(folderPath)
%LOOKSNORMALISED  True if every image sampled is exactly 320x320.
%   normalize_image writes a square targetSide image, so a folder of them is
%   uniform. Anything else - a mix of sizes, or one image that is not square -
%   has not been through the treatment.

    d = dir(folderPath);
    d = d(~[d.isdir]);
    names = {d.name}';
    if isempty(names)
        tf = false;
        return;
    end

    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    names = sort(names(isImage));
    if isempty(names)
        tf = false;
        return;
    end

    % Sampling the first few is enough: the question is whether this folder
    % came out of normalize_folder, and that writes every file the same size.
    nCheck = min(8, numel(names));
    tf = true;
    for i = 1:nCheck
        try
            info = imfinfo(fullfile(folderPath, names{i}));
            if info(1).Width ~= 320 || info(1).Height ~= 320
                tf = false;
                return;
            end
        catch
            tf = false;
            return;
        end
    end
end
