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
%   The variants added per image, each at a RANDOM quality rather than a fixed
%   one - every training image already exists at exactly one compression state,
%   which is why the model treats compression level as identity. Drawing at
%   random spreads the dataset across the whole quality scale instead of piling
%   it on two points:
%       _qhi      re-saved as JPEG, quality drawn from 70-95
%       _qlo      re-saved as JPEG, quality drawn from 40-70
%       _rweb     shrunk 60-90% then saved as JPEG 70-95 (a web pipeline)
%       _soft     shrunk by the same 60-90% as _rweb and scaled BACK UP,
%                 then JPEG 70-95 - the same detail loss with the pixel
%                 count preserved, so it survives on small images where
%                 _rweb cannot be written
%       _s##      THE SCALE LADDER: the same image at fixed detail levels,
%                 shrunk to ##% and scaled back up. Not random, and identical
%                 for every image in both classes.
%
%   WHY THE LADDER IS NOT JUST ANOTHER VARIANT
%
%   Normalisation now measures every image over the same fraction of itself,
%   but the images still ARRIVE at that fraction differently: a 6000px camera
%   file is downsampled 13x to get there and a 460px thumbnail not at all, so
%   the one that came from a big source is left carrying more detail per pixel.
%   That is the residue of the fault where a photograph's verdict tracked its
%   resolution - real under about 1200px, AI over about 1300px, whatever the
%   picture actually was.
%
%   The random _soft draw spreads detail level a little; a fixed ladder spreads
%   it deliberately and equally. Every photograph now appears in training at
%   four detail levels under one label, so "how much fine detail per pixel"
%   cannot separate the classes - there is no setting of it that is more real
%   or more AI. The rungs are deterministic rather than drawn at random so that
%   both classes get exactly the same ladder, which is the whole point: a
%   ladder that went deeper in one class would be a new confound in place of
%   the old one.
%
%   scale_sweep.m is the test that says whether it worked.
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
%   rather than compounded.
%
%   To undo, delete every file matching *_qhi.*, *_qlo.*, *_rweb.*, *_soft.*
%   and *_s??.*
%   (If an earlier run left *_q85.*, *_q60.* or *_r75q85.* files, delete those
%   too - they are recognised and not re-augmented, but they are redundant.)
%
%   BEFORE RUNNING: back up dataset_crop.csv and filenames_crop.txt. The next
%   extraction overwrites them, and the numbers already in your report came
%   from the current ones.

    if nargin < 1 || isempty(doResizeVariant)
        doResizeVariant = true;
    end

    Q_HI    = [70 95];      % light compression band
    Q_LO    = [40 70];       % heavy compression band
    R_RANGE = [0.60 0.90];   % web resize band
    MIN_AFTER = 256;         % below this the extractor upscales, a different effect

    % The scale ladder. Detail levels, not sizes: each rung shrinks by the
    % factor and puts the pixel count straight back, so a rung can be written
    % for every image in both classes whatever its dimensions - no rung is
    % ever skipped for being too small, which is what keeps the two classes'
    % ladders identical. 0.85 down to 0.40 covers roughly what a 1.2x and a
    % 2.5x downsample cost, which brackets the range real uploads span.
    LADDER = [0.85 0.70 0.55 0.40];

    rng(42);                 % reproducible draws

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
        isMade  = ~cellfun(@isempty, ...
                  regexpi(names, '_(qhi|qlo|rweb|soft|s\d+|q85|q60|r75q85)\.', 'once'));
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
                outHi = fullfile(folder, [base '_qhi.jpg']);
                outLo = fullfile(folder, [base '_qlo.jpg']);

                if isfile(outHi)
                    nExisted = nExisted + 1;
                else
                    imwrite(img, outHi, 'Quality', randQuality(Q_HI));
                    nMade = nMade + 1;
                end

                if isfile(outLo)
                    nExisted = nExisted + 1;
                else
                    imwrite(img, outLo, 'Quality', randQuality(Q_LO));
                    nMade = nMade + 1;
                end

                % Detail loss WITHOUT a dimension change. _rweb shrinks the
                % file, so on an already-normalised 320px set it lands under
                % MIN_AFTER and is skipped - which silently removed the one
                % variant that taught the model to ignore web laundering.
                % This one throws the same information away and puts the
                % pixel count back, so it survives at any size. It is also
                % the exact operation the extractor performs on a small
                % image: downscale happened somewhere, upscale to measure.
                outS = fullfile(folder, [base '_soft.jpg']);
                if isfile(outS)
                    nExisted = nExisted + 1;
                else
                    sFactor = R_RANGE(1) + rand * (R_RANGE(2) - R_RANGE(1));
                    shrunk  = imresize(img, sFactor);
                    restored = imresize(shrunk, [size(img, 1) size(img, 2)]);
                    imwrite(restored, outS, 'Quality', randQuality(Q_HI));
                    nMade = nMade + 1;
                end

                % The scale ladder, identical in both classes.
                for r = LADDER
                    outLad = fullfile(folder, ...
                                      sprintf('%s_s%02d.jpg', base, round(r * 100)));
                    if isfile(outLad)
                        nExisted = nExisted + 1;
                        continue;
                    end
                    shrunk   = imresize(img, r);
                    restored = imresize(shrunk, [size(img, 1) size(img, 2)]);
                    imwrite(restored, outLad, 'Quality', randQuality(Q_HI));
                    nMade = nMade + 1;
                end

                if doResizeVariant
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
    fprintf('  5. scale_sweep(''Dataset/Real_Images'')  then\n');
    fprintf('     python scale_sweep.py scale_sweep.csv\n');
    fprintf('     -> the ladder above is what this has to make flat: the same\n');
    fprintf('        photograph must score the same at every resolution.\n');

end


function q = randQuality(band)
%RANDQUALITY  An integer JPEG quality drawn uniformly from [band(1), band(2)].
    q = round(band(1) + rand * (band(2) - band(1)));
end
