function scale_sweep(srcDir, nSample, outCsv, rungs)
%SCALE_SWEEP  Does the verdict follow the RESOLUTION of the upload?
%
%   scale_sweep('Dataset/Real_Images')
%   scale_sweep('Dataset/AI_Images', 40)
%   scale_sweep(srcDir, 40, 'sweep_ai.csv')
%   scale_sweep(srcDir, 40, 'sweep.csv', [448 640 896 1280 1792])
%
%   Takes a sample of photographs and measures EACH ONE at several source
%   resolutions - the same picture, the same content, differing only in how
%   many pixels it is stored in. Every rung goes through the full pipeline the
%   live demo uses: normalize_image, then extractImageFeatures.
%
%   Nothing about a photograph's origin changes when it is resized. So a
%   detector that reads generation rather than provenance must return nearly
%   the same score at every rung, and scale_sweep.py checks exactly that:
%
%       matlab -batch "scale_sweep('Dataset/Real_Images', 40)"
%       python scale_sweep.py scale_sweep.csv
%
%   This is the test the project did not have. The fault it exists to catch -
%   images under about 1200px called real and over about 1300px called AI,
%   whatever they were - was invisible to every accuracy number in the record,
%   because each test image appeared at exactly one resolution and the two
%   classes' resolutions did not overlap. Accuracy on such a set is high
%   precisely BECAUSE the model is reading resolution.
%
%   Rungs are only ever produced by DOWNSCALING. An upscaled rung would be a
%   low-pass filtered image and would measure the upscaler, not the detector,
%   so an image simply contributes no rungs above its native short side.
%
%   Writes one row per (image, rung):
%
%       col 1      image index
%       col 2      rung: the short side the source was resized to
%       col 3      the image's native short side
%       col 4      preScale the normalisation then applied
%       col 5..234 the 230 features
%
%   plus <outCsv>.images.txt, row-aligned with the image indices.

    if nargin < 2 || isempty(nSample), nSample = 30;               end
    if nargin < 3 || isempty(outCsv),  outCsv  = 'scale_sweep.csv'; end
    if nargin < 4 || isempty(rungs)
        % Spans the range a person actually uploads from, and brackets the
        % 1200-1300px band where the verdict was observed to flip.
        rungs = [448 560 704 896 1120 1400 1760 2240];
    end

    rungs = sort(unique(rungs(:)'), 'descend');
    opts  = normalize_defaults();

    if strcmpi(opts.scaleMode, 'shortside')
        floorSide = opts.scaleSide;
    else
        floorSide = opts.cropSide;
    end
    rungs = rungs(rungs >= floorSide);
    if isempty(rungs)
        error('scale_sweep:noRungs', ...
              'Every rung is below the %dpx normalisation floor.', floorSide);
    end

    if ~isfolder(srcDir)
        error('scale_sweep:noFolder', 'Cannot find %s', srcDir);
    end

    names = imageNames(srcDir);
    if isempty(names)
        error('scale_sweep:noImages', 'No .jpg/.jpeg/.png images in %s', srcDir);
    end

    % Only images large enough to reach at least two rungs can say anything
    % about scale sensitivity: one rung is one measurement, not a sweep.
    rng(42);                                    % reproducible sample
    names = names(randperm(numel(names)));

    workDir = tempname;
    mkdir(workDir);
    cleanup = onCleanup(@() rmdir(workDir, 's'));

    fprintf('\n');
    fprintf('======================================================================\n');
    fprintf('  SCALE SWEEP - is the verdict a property of the picture or its size?\n');
    fprintf('======================================================================\n');
    fprintf('  source  : %s\n', srcDir);
    fprintf('  rungs   : %s  (short side, px)\n', mat2str(fliplr(rungs)));
    fprintf('  pipeline: normalize_image (%s) -> extractImageFeatures\n\n', ...
            opts.scaleMode);

    rows   = zeros(0, 234);
    used   = {};
    nTried = 0;

    for i = 1:numel(names)
        if numel(used) >= nSample
            break;
        end
        nTried = nTried + 1;
        src = fullfile(srcDir, names{i});

        try
            img = imread(src);
        catch err
            fprintf(2, '  %s: %s\n', names{i}, err.message);
            continue;
        end

        [h, w, ~] = size(img);
        native    = min(h, w);
        mine      = rungs(rungs <= native);     % downscale only
        if numel(mine) < 2
            continue;                            % too small to sweep
        end

        idx      = numel(used) + 1;
        theseRows = zeros(numel(mine), 234);
        okAll     = true;

        for k = 1:numel(mine)
            rung = mine(k);
            try
                rungPath = fullfile(workDir, sprintf('rung_%d.png', rung));
                writeRung(img, rung, rungPath);

                normPath = fullfile(workDir, sprintf('norm_%d.jpg', rung));
                [ok, info] = normalize_image(rungPath, normPath, opts);
                if ~ok
                    okAll = false;               % should not happen: rung >= floor
                    break;
                end

                feats = extractImageFeatures(normPath, 'crop');
                theseRows(k, :) = [idx, rung, native, info.preScale, feats];

            catch err
                fprintf(2, '  %s @ %dpx: %s\n', names{i}, rung, err.message);
                okAll = false;
                break;
            end
        end

        if okAll
            rows(end+1:end+numel(mine), :) = theseRows;  %#ok<AGROW>
            used{end+1} = names{i};                      %#ok<AGROW>
            fprintf('  %2d/%d  %-40s  native %dpx, %d rungs\n', ...
                    numel(used), nSample, names{i}, native, numel(mine));
        end
    end

    if isempty(used)
        error('scale_sweep:nothingUsable', ...
              ['No image in %s reached two rungs. Every rung is a DOWNSCALE, so ' ...
               'the\nsweep needs images of at least %dpx on the short side.'], ...
              srcDir, rungs(end-1));
    end

    writematrix(rows, outCsv);

    listPath = [outCsv '.images.txt'];
    fid = fopen(listPath, 'w');
    if fid == -1
        warning('scale_sweep:noList', 'Could not write %s', listPath);
    else
        c = onCleanup(@() fclose(fid));
        for i = 1:numel(used)
            fprintf(fid, '%d\t%s\n', i, used{i});
        end
        clear c;
    end

    fprintf('\n  %d images, %d rows -> %s\n', numel(used), size(rows, 1), outCsv);
    fprintf('  skipped %d that were too small to sweep\n', nTried - numel(used));
    fprintf('\n  Next:\n    python scale_sweep.py %s\n\n', outCsv);
end


%% ================================================================
%  Local functions
%  ================================================================

function names = imageNames(folderPath)
%IMAGENAMES  The image files in a folder, augmented copies excluded.
%   An augmented copy is a resized or recompressed version of an image already
%   in the folder; sweeping both would report the same photograph twice.

    d = dir(folderPath);
    d = d(~[d.isdir]);
    names = {d.name}';
    if isempty(names)
        return;
    end

    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    isMade  = ~cellfun(@isempty, ...
                       regexpi(names, '_(qhi|qlo|rweb|soft|s\d+|q85|q60|r75q85)\.', 'once'));
    names   = sort(names(isImage & ~isMade));
end


function writeRung(img, shortSide, dstPath)
%WRITERUNG  The same picture, stored at a different resolution.
%
%   Written as PNG on purpose. A JPEG rung would carry a second round of
%   compression whose strength depends on the rung's pixel count, so the sweep
%   would be measuring compression as well as scale and could not separate
%   them. PNG is lossless, so the only thing that differs between rungs is the
%   resolution - which is the whole point of the experiment.

    [h, w, ~] = size(img);
    if h <= w
        newSize = [shortSide, round(w * shortSide / h)];
    else
        newSize = [round(h * shortSide / w), shortSide];
    end

    imwrite(imresize(img, newSize, 'bicubic'), dstPath);
end
