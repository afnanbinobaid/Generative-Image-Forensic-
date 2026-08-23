function make_variants(nSample, outRoot)
%MAKE_VARIANTS  Build the controlled edit experiment (diagnostic D3).
%
%   make_variants              100 images into .\variants
%   make_variants(150)         150 images
%   make_variants(100, 'v2')   into .\v2
%
%   Takes a sample of Dataset/Real_Images and writes several edited copies of
%   each into its own folder. Every copy is the SAME photograph differing by
%   exactly one edit, so whatever changes in the detector's verdict must be
%   caused by that edit and nothing else.
%
%     control    unchanged pixels           (flip rate here must be ~0)
%     resize75   shrunk to 75%
%     jpeg85     re-saved as JPEG quality 85
%     jpeg60     re-saved as JPEG quality 60
%     denoise    slight blur, imgaussfilt(0.6)
%     sharpen    imsharpen
%     web_sim    resize75 + sharpen + JPEG 85, a typical stock-site pipeline
%
%   Everything except the JPEG variants is written as PNG. PNG is lossless, so
%   those folders isolate their one edit instead of also adding a second round
%   of JPEG compression - which would confound the whole experiment.
%
%   Only images at least 350px on the short side are sampled, so the 75% copies
%   stay above the 256px crop window. Below that the extractor has to scale the
%   image up, which is a different effect entirely and would pollute the result.
%
%   Then, for each folder:
%       extract_folder('variants\sharpen', 'sharpen.csv')
%       python score_folder.py sharpen.csv real

    if nargin < 1 || isempty(nSample),  nSample = 100;        end
    if nargin < 2 || isempty(outRoot),  outRoot = 'variants'; end

    MIN_SHORT_SIDE = 350;   % so a 75% copy stays above the 256px crop window
    RESIZE_FACTOR  = 0.75;

    srcDir = fullfile(pwd, 'Dataset', 'Real_Images');
    if ~isfolder(srcDir)
        error('make_variants:noSource', 'Cannot find %s', srcDir);
    end

    d = dir(srcDir);
    d = d(~[d.isdir]);
    names = {d.name}';
    if ~isempty(names)
        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        names   = sort(names(isImage));
    end
    if isempty(names)
        error('make_variants:noImages', 'No images in %s', srcDir);
    end

    % Keep only images big enough that a 75% copy still clears the crop window.
    fprintf('Checking sizes of %d candidates...\n', numel(names));
    keep = false(numel(names), 1);
    for i = 1:numel(names)
        try
            info = imfinfo(fullfile(srcDir, names{i}));
            shortSide = min(info(1).Width, info(1).Height);
            keep(i) = shortSide * RESIZE_FACTOR >= 260 && shortSide >= MIN_SHORT_SIDE;
        catch
            keep(i) = false;
        end
    end
    names = names(keep);

    if isempty(names)
        error('make_variants:tooSmall', ...
              ['No images are at least %dpx on the short side. The resize ' ...
               'variant cannot be tested on this dataset.'], MIN_SHORT_SIDE);
    end
    fprintf('%d images are large enough to use.\n', numel(names));

    rng(42);
    if numel(names) > nSample
        names = names(sort(randperm(numel(names), nSample)));
    end
    fprintf('Sampling %d of them.\n\n', numel(names));

    variants = {'control', 'resize75', 'jpeg85', 'jpeg60', 'denoise', ...
                'sharpen', 'web_sim'};
    for v = 1:numel(variants)
        folder = fullfile(outRoot, variants{v});
        if ~isfolder(folder)
            mkdir(folder);
        end
    end

    nDone = 0;
    nSkip = 0;

    for i = 1:numel(names)
        srcFile = fullfile(srcDir, names{i});
        try
            img = imread(srcFile);
            if size(img, 3) == 1
                img = repmat(img, [1 1 3]);
            elseif size(img, 3) > 3
                img = img(:,:,1:3);
            end

            [~, base] = fileparts(names{i});
            small     = imresize(img, RESIZE_FACTOR);

            % PNG is lossless, so these folders test their one edit only.
            imwrite(img,                fullfile(outRoot, 'control',  [base '.png']));
            imwrite(small,              fullfile(outRoot, 'resize75', [base '.png']));
            imwrite(imgaussfilt(img, 0.6), fullfile(outRoot, 'denoise', [base '.png']));
            imwrite(imsharpen(img),     fullfile(outRoot, 'sharpen',  [base '.png']));

            % These two exist to test re-compression, so they must be JPEG.
            imwrite(img, fullfile(outRoot, 'jpeg85', [base '.jpg']), 'Quality', 85);
            imwrite(img, fullfile(outRoot, 'jpeg60', [base '.jpg']), 'Quality', 60);

            % The realistic combination.
            imwrite(imsharpen(small), fullfile(outRoot, 'web_sim', [base '.jpg']), ...
                    'Quality', 85);

            nDone = nDone + 1;
        catch ME
            nSkip = nSkip + 1;
            fprintf(2, 'SKIPPED: %s  (%s)\n', names{i}, ME.message);
        end

        if mod(i, 25) == 0
            fprintf('  %d / %d\n', i, numel(names));
        end
    end

    fprintf('\nWrote %d images per variant into %s (%d skipped)\n', ...
            nDone, fullfile(pwd, outRoot), nSkip);
    fprintf('\nNext, for each variant:\n');
    for v = 1:numel(variants)
        fprintf('    extract_folder(''%s'', ''%s.csv'')\n', ...
                fullfile(outRoot, variants{v}), variants{v});
    end
    fprintf('\nthen in a terminal:\n');
    fprintf('    python score_folder.py control.csv real     <- expect ~0%% flipped\n');
    fprintf('    python score_folder.py sharpen.csv real     <- and so on\n');
end
