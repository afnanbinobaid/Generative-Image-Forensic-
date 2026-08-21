function extract_folder(folderPath, outCsv)
%EXTRACT_FOLDER  Measure every image in a folder into a feature CSV.
%
%   extract_folder('E:\path\to\new_images', 'new.csv')
%
%   For testing the detector against a generator it has never seen. Writes 230
%   feature columns per image (no label) plus a row-aligned <outCsv>.filenames.txt,
%   then score it in Python:
%
%       python score_folder.py new.csv ai
%
%   Uses extractImageFeatures(), the same measurement code as the training set.

    if nargin < 2 || isempty(outCsv)
        outCsv = 'folder_features.csv';
    end
    if ~isfolder(folderPath)
        error('extract_folder:noFolder', 'Not a folder: %s', folderPath);
    end

    d = dir(folderPath);
    d = d(~[d.isdir]);
    names = {d.name}';
    if ~isempty(names)
        isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
        names = sort(names(isImage));
    end
    if isempty(names)
        error('extract_folder:noImages', ...
              'No .jpg/.jpeg/.png images in %s', folderPath);
    end

    fprintf('Measuring %d images from %s\n', numel(names), folderPath);

    matrix = NaN(numel(names), 230);
    kept   = cell(numel(names), 1);
    nKept  = 0;
    nSkip  = 0;

    for i = 1:numel(names)
        f = fullfile(folderPath, names{i});
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

    fprintf('\nMeasured %d, skipped %d\n', nKept, nSkip);
    fprintf('Wrote %s (%d x %d)\n', fullfile(pwd, outCsv), nKept, 230);
    fprintf('\nNow score it:\n    python score_folder.py %s ai\n', outCsv);
end
