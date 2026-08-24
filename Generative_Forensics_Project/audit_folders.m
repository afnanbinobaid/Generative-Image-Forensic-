function audit_folders(folderA, folderB, labelA, labelB)
%AUDIT_FOLDERS  Compare two folders' native image properties before trusting a score.
%
%   audit_folders('E:\test\real', 'E:\test\ai')
%   audit_folders('E:\test\real', 'E:\test\ai', 'real', 'midjourney')
%
%   Section 4 of the project record records the most expensive lesson here: the
%   two classes were not comparable BEFORE measurement, and resizing them by
%   different factors manufactured a high-frequency difference that had nothing
%   to do with whether an image was generated. That was caught by auditing the
%   raw inputs rather than the outputs, and this is that audit as a reusable
%   script.
%
%   Run it on any new pair of folders before believing an accuracy from them.
%   What it reports, and why each matters:
%
%     native dimensions   If one class is systematically larger, a fixed crop
%                         covers a different fraction of the frame, and any
%                         resize applied by whoever built the dataset will have
%                         low-pass filtered the two classes by different amounts.
%
%     bytes per pixel     A proxy for JPEG quality. The detector keys on fine
%                         detail, so if one class was saved at a lower quality
%                         than the other, that alone can decide the verdict -
%                         see sections 7 to 9.
%
%     below the crop      Images smaller than 256px on a side are upscaled by
%                         the extractor, which destroys high-frequency energy.
%                         A class with many such images will look generated.
%
%     format mix          If one class is PNG and the other JPEG, the classifier
%                         can learn the container instead of the content.
%
%   A large gap on any row means an accuracy measured on these folders is
%   partly measuring the gap, not the detector.

    if nargin < 3 || isempty(labelA), labelA = 'folder A'; end
    if nargin < 4 || isempty(labelB), labelB = 'folder B'; end

    A = collectStats(folderA, labelA);
    B = collectStats(folderB, labelB);

    fprintf('\n');
    fprintf('==================================================================\n');
    fprintf('FOLDER AUDIT - are these two folders comparable before measurement?\n');
    fprintf('==================================================================\n\n');

    fprintf('%-26s%18s%18s\n', '', labelA, labelB);
    fprintf('%s\n', repmat('-', 1, 62));

    row('images readable',      A.n,               B.n,            '%18d');
    row('median width',         A.medW,            B.medW,         '%18.0f');
    row('median height',        A.medH,            B.medH,         '%18.0f');
    row('mean width',           A.meanW,           B.meanW,        '%18.1f');
    row('mean height',          A.meanH,           B.meanH,        '%18.1f');
    row('min short side',       A.minShort,        B.minShort,     '%18d');
    row('max short side',       A.maxShort,        B.maxShort,     '%18d');
    row('below 256px (upscaled)', A.nSmall,        B.nSmall,       '%18d');
    row('median bytes/pixel',   A.medBpp,          B.medBpp,       '%18.4f');
    row('median file size (kB)',A.medKB,           B.medKB,        '%18.1f');
    row('JPEG files',           A.nJpg,            B.nJpg,         '%18d');
    row('PNG files',            A.nPng,            B.nPng,         '%18d');

    fprintf('\n');

    %% ------------------------------------------------------------ verdicts
    fprintf('Reading the table\n');
    fprintf('%s\n', repmat('-', 1, 62));

    flagged = false;

    if A.medShort > 0 && B.medShort > 0
        ratio = max(A.medShort, B.medShort) / min(A.medShort, B.medShort);
        if ratio >= 1.25
            flagged = true;
            if A.medShort > B.medShort
                bigName = labelA; smallName = labelB;
            else
                bigName = labelB; smallName = labelA;
            end
            fprintf(2, ['  RESOLUTION GAP %.2fx: %s images are typically much larger\n' ...
                        '    than %s ones. Whatever resize either class received will\n' ...
                        '    have removed different amounts of fine detail, which is\n' ...
                        '    exactly the confound section 4 documents.\n'], ...
                    ratio, bigName, smallName);
        end
    end

    if A.medBpp > 0 && B.medBpp > 0
        bppRatio = max(A.medBpp, B.medBpp) / min(A.medBpp, B.medBpp);
        if bppRatio >= 1.35
            flagged = true;
            if A.medBpp > B.medBpp
                hi = labelA; lo = labelB;
            else
                hi = labelB; lo = labelA;
            end
            fprintf(2, ['  COMPRESSION GAP %.2fx: %s images carry far more bytes per\n' ...
                        '    pixel than %s ones, so they were probably saved at a higher\n' ...
                        '    JPEG quality. The detector reads missing fine detail as\n' ...
                        '    generation, so this alone can decide the verdict.\n'], ...
                    bppRatio, hi, lo);
        end
    end

    if A.nSmall > 0.05 * max(A.n,1) || B.nSmall > 0.05 * max(B.n,1)
        flagged = true;
        fprintf(2, ['  UPSCALING: %d of %d %s images and %d of %d %s images are below\n' ...
                    '    256px and will be scaled UP before measurement, destroying the\n' ...
                    '    high-frequency energy the detector depends on.\n'], ...
                A.nSmall, A.n, labelA, B.nSmall, B.n, labelB);
    end

    if (A.nPng > 0) ~= (B.nPng > 0) || (A.nJpg > 0) ~= (B.nJpg > 0)
        flagged = true;
        fprintf(2, ['  FORMAT MISMATCH: the two folders do not use the same file\n' ...
                    '    formats. A classifier can learn the container rather than the\n' ...
                    '    content - the record rules this out for the training set for\n' ...
                    '    exactly this reason.\n']);
    end

    if ~flagged
        fprintf('  No large gap found on any row. The two folders look comparable,\n');
        fprintf('  so an accuracy measured on them is more likely to be measuring\n');
        fprintf('  the detector rather than a difference in how they were prepared.\n');
    else
        fprintf('\n');
        fprintf(2, ['  At least one gap above is large enough to influence the verdict\n' ...
                    '  on its own. Treat any accuracy from these folders as partly a\n' ...
                    '  measurement of that gap until it is ruled out.\n']);
    end
    fprintf('\n');
end


function row(name, a, b, fmt)
    fprintf(['%-26s' fmt fmt '\n'], name, a, b);
end




function S = collectStats(folderPath, label)
%COLLECTSTATS  Native properties of every readable image in a folder.

    if ~isfolder(folderPath)
        error('audit_folders:noFolder', 'Not a folder: %s', folderPath);
    end

    d = dir(folderPath);
    d = d(~[d.isdir]);
    names = {d.name}';
    if ~isempty(names)
        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        names = sort(names(isImage));
    end
    if isempty(names)
        error('audit_folders:noImages', 'No .jpg/.jpeg/.png images in %s', folderPath);
    end

    fprintf('Reading %d images from %s ...\n', numel(names), label);

    W = []; H = []; BPP = []; KB = [];
    nJpg = 0; nPng = 0; nSkip = 0;

    for i = 1:numel(names)
        f = fullfile(folderPath, names{i});
        try
            info = imfinfo(f);
            info = info(1);
            W(end+1)   = info.Width;                          %#ok<AGROW>
            H(end+1)   = info.Height;                         %#ok<AGROW>
            KB(end+1)  = info.FileSize / 1024;                %#ok<AGROW>
            BPP(end+1) = info.FileSize / (info.Width * info.Height);  %#ok<AGROW>
            if strcmpi(info.Format, 'png')
                nPng = nPng + 1;
            else
                nJpg = nJpg + 1;
            end
        catch
            nSkip = nSkip + 1;
        end
    end

    if nSkip > 0
        fprintf(2, '  (%d unreadable, skipped)\n', nSkip);
    end

    short = min(W, H);

    S = struct( ...
        'n',        numel(W), ...
        'medW',     median(W), ...
        'medH',     median(H), ...
        'meanW',    mean(W), ...
        'meanH',    mean(H), ...
        'medShort', median(short), ...
        'minShort', min(short), ...
        'maxShort', max(short), ...
        'nSmall',   sum(short < 256), ...
        'medBpp',   median(BPP), ...
        'medKB',    median(KB), ...
        'nJpg',     nJpg, ...
        'nPng',     nPng);
end
