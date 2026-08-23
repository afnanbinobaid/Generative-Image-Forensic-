function make_augmented(doResizeVariant)
%MAKE_AUGMENTED  Add compressed copies of every training image, to BOTH classes.
%
%   make_augmented          adds three variants per image
%   make_augmented(false)   adds only the two JPEG variants (faster)
%
%   The controlled experiment showed JPEG re-compression alone flips 81% of real
%   photographs to "AI": compression removes fine detail, and the detector reads
%   missing fine detail as generation. The fix is to show it compressed images
%   during training so compression stops being informative.
%
%   The variants added per image:
%       _q85      re-saved as JPEG quality 85
%       _q60      re-saved as JPEG quality 60
%       _r75q85   shrunk to 75% then saved as JPEG 85 (a realistic web pipeline)
%
%   CRITICAL: these are added to Real_Images AND AI_Images alike. Augmenting only
%   the real images would teach the model that "compressed means real", which is
%   a worse failure than the one being fixed - it would score well on a test set
%   and be useless in practice. Both classes, identical treatment, always.
%
%   Files are written alongside the originals, so feature_extractor.m needs no
%   changes. Running this twice is safe: already-augmented files are skipped
%   rather than compounded.
%
%   To undo, delete every file matching *_q85.*, *_q60.* and *_r75q85.*
%
%   BEFORE RUNNING: back up dataset_crop.csv and filenames_crop.txt. The next
%   extraction overwrites them, and the numbers already in your report came
%   from the current ones.

    if nargin < 1 || isempty(doResizeVariant)
        doResizeVariant = true;
    end

    RESIZE_FACTOR = 0.75;
    MIN_AFTER     = 256;    % below this the extractor upscales, a different effect

    folders = {fullfile(pwd, 'Dataset', 'Real_Images'), ...
               fullfile(pwd, 'Dataset', 'AI_Images')};
    labels  = {'Real_Images', 'AI_Images'};

    for f = 1:numel(folders)
        if ~isfolder(folders{f})
            error('make_augmented:noFolder', 'Cannot find %s', folders{f});
        end
    end

    fprintf('Augmenting BOTH classes so compression carries no label information.\n\n');

    totals = zeros(numel(folders), 1);

    for f = 1:numel(folders)
        folder = folders{f};

        d = dir(folder);
        d = d(~[d.isdir]);
        names = {d.name}';
        if isempty(names)
            fprintf('%s: empty, skipping\n', labels{f});
            continue;
        end

        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        % Never augment an augmented file - that would stack compression.
        isMade  = ~cellfun(@isempty, regexpi(names, '_(q85|q60|r75q85)\.', 'once'));
        names   = sort(names(isImage & ~isMade));

        fprintf('%s: %d original images\n', labels{f}, numel(names));

        nMade    = 0;
        nExisted = 0;
        nSkip    = 0;
        nNoResize = 0;

        for i = 1:numel(names)
            src = fullfile(folder, names{i});
            try
                img = imread(src);
                if size(img, 3) == 1
                    img = repmat(img, [1 1 3]);
                elseif size(img, 3) > 3
                    img = img(:,:,1:3);
                end

                [~, base] = fileparts(names{i});

                % Skip outputs that already exist, so re-running after an
                % interruption resumes instead of redoing hours of work.
                out85 = fullfile(folder, [base '_q85.jpg']);
                out60 = fullfile(folder, [base '_q60.jpg']);

                if isfile(out85)
                    nExisted = nExisted + 1;
                else
                    imwrite(img, out85, 'Quality', 85);
                    nMade = nMade + 1;
                end

                if isfile(out60)
                    nExisted = nExisted + 1;
                else
                    imwrite(img, out60, 'Quality', 60);
                    nMade = nMade + 1;
                end

                if doResizeVariant
                    outR = fullfile(folder, [base '_r75q85.jpg']);
                    if isfile(outR)
                        nExisted = nExisted + 1;
                    else
                        small = imresize(img, RESIZE_FACTOR);
                        if min(size(small, 1), size(small, 2)) >= MIN_AFTER
                            imwrite(small, outR, 'Quality', 85);
                            nMade = nMade + 1;
                        else
                            nNoResize = nNoResize + 1;
                        end
                    end
                end

            catch ME
                nSkip = nSkip + 1;
                fprintf(2, '  SKIPPED %s  (%s)\n', names{i}, ME.message);
            end

            if mod(i, 250) == 0
                fprintf('  %d / %d\n', i, numel(names));
            end
        end

        totals(f) = numel(names) + nMade + nExisted;
        fprintf('  wrote %d new files', nMade);
        if nExisted > 0
            fprintf(', %d already existed and were left alone', nExisted);
        end
        if nNoResize > 0
            fprintf(', %d too small for the resize variant', nNoResize);
        end
        if nSkip > 0
            fprintf(', %d unreadable', nSkip);
        end
        fprintf('\n  %s now holds %d images\n\n', labels{f}, totals(f));
    end

    ratio = 0;
    if min(totals) > 0
        ratio = max(totals) / min(totals);
    end

    fprintf('=========================================================\n');
    fprintf('Real_Images : %d\n', totals(1));
    fprintf('AI_Images   : %d\n', totals(2));
    if ratio > 1.15
        fprintf(2, ['\nWARNING: the classes are now imbalanced %.2f:1. Both were\n' ...
                    'augmented the same way, so this reflects a size difference in\n' ...
                    'the originals. Consider class_weight when training.\n'], ratio);
    else
        fprintf('Balance     : %.2f:1  (both classes augmented identically)\n', ratio);
    end
    fprintf('=========================================================\n');
    fprintf('\nNext:\n');
    fprintf('  1. back up dataset_crop.csv and filenames_crop.txt if you have not\n');
    fprintf('  2. feature_extractor          (re-measures everything - this is slow)\n');
    fprintf('  3. python train_model.py\n');
    fprintf('  4. re-run the diagnostics to confirm the fix worked\n');
end
