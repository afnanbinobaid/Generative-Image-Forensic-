function make_augmented(nVariants, doResizeVariant)
%MAKE_AUGMENTED  Add compressed copies of every training image, to BOTH classes.
%
%   make_augmented                 all 3 recipes/image (qhi+qlo+rweb) -> 4x the dataset
%   make_augmented(1)              1 random recipe/image              -> 2x the dataset
%   make_augmented(2)              2 random recipes/image             -> 3x the dataset
%   make_augmented(3, false)       qhi+qlo only, rweb excluded from the pool
%
%   NVARIANTS controls how many of the three augmentation recipes - qhi
%   (light JPEG, quality 70-95), qlo (heavy JPEG, quality 40-70), and rweb
%   (resize 60-90% then JPEG 70-95) - are applied to each image. Recipes are
%   chosen at random PER IMAGE, without replacement, rather than the same
%   fixed subset for every image, so the full compression range still shows
%   up across the dataset even when NVARIANTS is small.
%
%   Default is 3 (every recipe, every image): the original behaviour, taking
%   a 10,000-image dataset to 40,000. Before committing to that ~2-hour
%   extraction, it is worth checking whether it earns its cost: NVARIANTS=1
%   still gives every image at least one compressed sibling, which is what
%   actually breaks the compression-level-as-identity shortcut (section 8) -
%   fewer copies just means less redundancy per individual photo, not a
%   different mechanism. Test both on a small stratified subsample first
%   (make_augmented on a copy of Dataset with ~500-1000 images, extract,
%   train, score web_reals.csv) rather than assuming either answer.
%
%   NOTE: this used to be make_augmented(doResizeVariant). The first
%   argument is now NVARIANTS - call make_augmented(3, false) for the old
%   make_augmented(false) behaviour.
%
%   The controlled experiment showed JPEG re-compression alone flips 81% of real
%   photographs to "AI": compression removes fine detail, and the detector reads
%   missing fine detail as generation. The fix is to show it compressed images
%   during training so compression stops being informative.
%
%   Re-saving an already-JPEG image double-compresses it. That is deliberate:
%   web photographs are almost always compressed more than once - saved by the
%   photographer, re-encoded on upload, often again per display size.
%
%   CRITICAL: these are added to Real_Images AND AI_Images alike. Augmenting only
%   the real images would teach the model that "compressed means real", which is
%   a worse failure than the one being fixed - it would score well on a test set
%   and be useless in practice. Both classes, identical treatment, always.
%
%   Files are written alongside the originals, so feature_extractor.m needs no
%   changes. Running this twice is safe: already-augmented files are skipped
%   rather than compounded, and the fixed seed reproduces the same per-image
%   recipe choices, so an interrupted run resumes cleanly.
%
%   To undo, delete every file matching *_qhi.*, *_qlo.* and *_rweb.*
%   (If an earlier run left *_q85.*, *_q60.* or *_r75q85.* files, delete those
%   too - they are recognised and not re-augmented, but they are redundant.)
%
%   BEFORE RUNNING: back up dataset_crop.csv and filenames_crop.txt. The next
%   extraction overwrites them, and the numbers already in your report came
%   from the current ones.

    if nargin < 1 || isempty(nVariants)
        nVariants = 3;
    end
    if nargin < 2 || isempty(doResizeVariant)
        doResizeVariant = true;
    end

    Q_HI    = [70 95];      % light compression band
    Q_LO    = [40 70];       % heavy compression band
    R_RANGE = [0.60 0.90];   % web resize band
    MIN_AFTER = 256;         % below this the extractor upscales, a different effect

    rng(42);                 % reproducible draws

    recipePool = {'qhi', 'qlo'};
    if doResizeVariant
        recipePool{end+1} = 'rweb';
    end
    nVariants = min(nVariants, numel(recipePool));

    folders = {fullfile(pwd, 'Dataset', 'Real_Images'), ...
               fullfile(pwd, 'Dataset', 'AI_Images')};
    labels  = {'Real_Images', 'AI_Images'};

    for f = 1:numel(folders)
        if ~isfolder(folders{f})
            error('make_augmented:noFolder', 'Cannot find %s', folders{f});
        end
    end

    fprintf('Augmenting BOTH classes so compression carries no label information.\n');
    fprintf('%d of %d recipes per image, chosen at random.\n\n', ...
            nVariants, numel(recipePool));

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
        isMade  = ~cellfun(@isempty, regexpi(names, '_(qhi|qlo|rweb|q85|q60|r75q85)\.', 'once'));
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

                % Which recipes this particular image gets, drawn without
                % replacement so a partial NVARIANTS still spreads across the
                % whole pool over the dataset instead of always dropping the
                % same recipe.
                chosen = recipePool(randperm(numel(recipePool), nVariants));

                % Skip outputs that already exist, so re-running after an
                % interruption resumes instead of redoing hours of work.
                if ismember('qhi', chosen)
                    outHi = fullfile(folder, [base '_qhi.jpg']);
                    if isfile(outHi)
                        nExisted = nExisted + 1;
                    else
                        imwrite(img, outHi, 'Quality', randQuality(Q_HI));
                        nMade = nMade + 1;
                    end
                end

                if ismember('qlo', chosen)
                    outLo = fullfile(folder, [base '_qlo.jpg']);
                    if isfile(outLo)
                        nExisted = nExisted + 1;
                    else
                        imwrite(img, outLo, 'Quality', randQuality(Q_LO));
                        nMade = nMade + 1;
                    end
                end

                if ismember('rweb', chosen)
                    outR = fullfile(folder, [base '_rweb.jpg']);
                    if isfile(outR)
                        nExisted = nExisted + 1;
                    else
                        factor = R_RANGE(1) + rand * (R_RANGE(2) - R_RANGE(1));
                        small  = imresize(img, factor);
                        if min(size(small, 1), size(small, 2)) >= MIN_AFTER
                            imwrite(small, outR, 'Quality', randQuality(Q_HI));
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


function q = randQuality(band)
%RANDQUALITY  An integer JPEG quality drawn uniformly from [band(1), band(2)].
    q = round(band(1) + rand * (band(2) - band(1)));
end
