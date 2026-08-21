function score_folder(folderPath, groundTruth)
%SCORE_FOLDER  Run the trained detector over every image in a folder.
%
%   score_folder('C:\path\to\images')          predictions only
%   score_folder('C:\path\to\images', 'ai')    folder is known to be generated
%   score_folder('C:\path\to\images', 'real')  folder is known to be photographs
%
%   Use this to test the detector against a generator it has never seen. None
%   of these images took part in training, so the accuracy it reports is a
%   clean generalisation number rather than an in-distribution one.
%
%   Writes score_<folder>.csv with one row per image: filename, score, verdict.
%
%   Run train_model.m first.

    MODEL_PATH = 'model.mat';

    if nargin < 2
        groundTruth = '';
    end
    groundTruth = lower(groundTruth);
    if ~isempty(groundTruth) && ~ismember(groundTruth, {'ai', 'real'})
        error('score_folder:badTruth', ...
              'Second argument must be ''ai'', ''real'', or omitted.');
    end
    if ~isfile(MODEL_PATH)
        error('score_folder:noModel', ...
              'Cannot find %s - run train_model.m first.', ...
              fullfile(pwd, MODEL_PATH));
    end
    if ~isfolder(folderPath)
        error('score_folder:noFolder', 'Not a folder: %s', folderPath);
    end

    S = load(MODEL_PATH);

    d = dir(folderPath);
    d = d(~[d.isdir]);
    if isempty(d)
        error('score_folder:empty', 'No files in %s', folderPath);
    end
    names   = {d.name}';
    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    names   = sort(names(isImage));
    if isempty(names)
        error('score_folder:noImages', ...
              'No .jpg/.jpeg/.png images in %s', folderPath);
    end

    fprintf('Scoring %d images from %s\n', numel(names), folderPath);
    fprintf('Model threshold: %.4f\n\n', S.threshold);

    scores  = NaN(numel(names), 1);
    kept    = cell(numel(names), 1);
    nKept   = 0;
    nSmall  = 0;
    nSkip   = 0;

    for i = 1:numel(names)
        f = fullfile(folderPath, names{i});
        try
            info = imfinfo(f);
            if info(1).Width < 256 || info(1).Height < 256
                nSmall = nSmall + 1;
            end

            features    = extractImageFeatures(f, 'crop');
            [~, sc]     = predict(S.model, features);
            nKept       = nKept + 1;
            scores(nKept) = sc(2);
            kept{nKept}   = names{i};
        catch ME
            nSkip = nSkip + 1;
            fprintf(2, 'SKIPPED: %s  (%s)\n', names{i}, ME.message);
        end
        if mod(i, 50) == 0
            fprintf('  %d / %d\n', i, numel(names));
        end
    end

    scores = scores(1:nKept);
    kept   = kept(1:nKept);
    predAI = scores >= S.threshold;

    fprintf('\n==========================================================\n');
    fprintf('Scored          : %d', nKept);
    if nSkip > 0
        fprintf('   (skipped %d)', nSkip);
    end
    fprintf('\n');
    if nSmall > 0
        fprintf('Below 256px     : %d  <- upscaled first, so unreliable\n', nSmall);
    end
    fprintf('Predicted AI    : %d  (%.1f%%)\n', sum(predAI),  100*mean(predAI));
    fprintf('Predicted real  : %d  (%.1f%%)\n', sum(~predAI), 100*mean(~predAI));
    fprintf('Score mean/med  : %.3f / %.3f\n', mean(scores), median(scores));
    fprintf('Score range     : %.3f - %.3f\n', min(scores), max(scores));

    if ~isempty(groundTruth)
        if strcmp(groundTruth, 'ai')
            correct = predAI;
        else
            correct = ~predAI;
        end
        fprintf('\nGround truth    : all %s\n', upper(groundTruth));
        fprintf('Accuracy        : %.1f%%  (%d / %d)\n', ...
                100*mean(correct), sum(correct), numel(correct));
        fprintf(['\nNo image here took part in training, so this is a clean\n' ...
                 'generalisation figure.\n']);
    end

    [~, folderName] = fileparts(strip_trailing_sep(folderPath));
    outPath = sprintf('score_%s.csv', folderName);
    fid = fopen(outPath, 'w');
    if fid == -1
        warning('score_folder:cannotWrite', 'Could not write %s', outPath);
        return;
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'filename,score,verdict\n');
    for i = 1:nKept
        if scores(i) >= S.threshold
            verdict = 'AI';
        else
            verdict = 'real';
        end
        fprintf(fid, '%s,%.6f,%s\n', kept{i}, scores(i), verdict);
    end
    clear cleanup;

    fprintf('\nPer-image results: %s\n', fullfile(pwd, outPath));
end


function p = strip_trailing_sep(p)
%STRIP_TRAILING_SEP  So fileparts sees the folder name, not an empty tail.
    while ~isempty(p) && (p(end) == '/' || p(end) == '\')
        p(end) = [];
    end
end
