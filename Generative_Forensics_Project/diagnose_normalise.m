function diagnose_normalise(folderPath, nCheck)
%DIAGNOSE_NORMALISE  Find out which step of the normalisation is failing.
%
%   diagnose_normalise('E:\path\to\folder')
%   diagnose_normalise(folder, 3)
%
%   normalize_folder reports a failure per image and moves on, which is right
%   for a batch run and useless for finding out why. This runs the same four
%   steps one at a time on the first few images, printing what each one
%   produced and stopping at the first one that throws - with the full stack,
%   so the failing line is named rather than guessed at.

    if nargin < 2 || isempty(nCheck), nCheck = 3; end

    fprintf('MATLAB %s\n', version);
    fprintf('Image Processing Toolbox: %s\n\n', tf(license('test', 'Image_Toolbox')));

    % Anything shadowing these would break every image identically, which is
    % the shape of the failure being chased.
    for fn = {'imread', 'imresize', 'rgb2ycbcr', 'ycbcr2rgb', 'imwrite'}
        w = which(fn{1});
        fprintf('  %-10s -> %s\n', fn{1}, w);
    end
    fprintf('\n');

    d = dir(folderPath);
    d = d(~[d.isdir]);
    names = {d.name}';
    isImage = ~cellfun(@isempty, regexpi(names, '\.(jpe?g|png)$', 'once'));
    names = sort(names(isImage));
    if isempty(names)
        error('No images in %s', folderPath);
    end

    for i = 1:min(nCheck, numel(names))
        src = fullfile(folderPath, names{i});
        fprintf('=== %s\n', names{i});

        try
            info = imfinfo(src);
            fprintf('  imfinfo : %s  %dx%d  %s  %d bit\n', ...
                    info(1).Format, info(1).Width, info(1).Height, ...
                    info(1).ColorType, info(1).BitDepth);
        catch err
            fprintf(2, '  imfinfo FAILED: %s\n', err.message);
        end

        try
            [raw, map] = imread(src);
            fprintf('  imread  : %s  size %s  colormap %s\n', ...
                    class(raw), mat2str(size(raw)), tf(~isempty(map)));
        catch err
            report('imread', err); continue;
        end

        try
            img = forceRgb(raw, map);
            fprintf('  to RGB  : %s  size %s\n', class(img), mat2str(size(img)));
        catch err
            report('toUint8Rgb', err); continue;
        end

        [h, w, ~] = size(img);
        if h < 448 || w < 448
            fprintf('  (under the 448 floor - would be dropped, not failed)\n\n');
            continue;
        end

        try
            crop = img(1:448, 1:448, :);
            small = imresize(crop, [320 320], 'bicubic');
            fprintf('  resize  : %s  size %s\n', class(small), mat2str(size(small)));
        catch err
            report('imresize', err); continue;
        end

        try
            ycc = rgb2ycbcr(small);
            fprintf('  rgb2ycbcr : size %s\n', mat2str(size(ycc)));
        catch err
            report('rgb2ycbcr', err); continue;
        end

        try
            cb = imresize(imresize(ycc(:,:,2), 0.5, 'box'), [320 320], 'bilinear');
            fprintf('  chroma  : size %s\n', mat2str(size(cb)));
        catch err
            report('chroma imresize', err); continue;
        end

        try
            out = ycbcr2rgb(cat(3, ycc(:,:,1), cb, cb));
            fprintf('  ycbcr2rgb : %s  size %s\n', class(out), mat2str(size(out)));
        catch err
            report('ycbcr2rgb', err); continue;
        end

        try
            tmp = [tempname '.jpg'];
            imwrite(out, tmp, 'Quality', 85);
            delete(tmp);
            fprintf('  imwrite : ok\n');
        catch err
            report('imwrite', err); continue;
        end

        fprintf('  ALL STEPS PASSED on this image\n');
        fprintf('\n');
    end

    fprintf('\nNow the real thing, with the stack:\n');
    src = fullfile(folderPath, names{1});
    try
        dst = [tempname '.jpg'];
        normalize_image(src, dst);
        if isfile(dst), delete(dst); end
        fprintf('  normalize_image succeeded on %s\n', names{1});
    catch err
        fprintf(2, '  normalize_image FAILED: %s\n', err.message);
        fprintf(2, '  identifier: %s\n', err.identifier);
        for k = 1:numel(err.stack)
            fprintf(2, '    in %s at line %d\n', err.stack(k).name, err.stack(k).line);
        end
    end
end


function s = tf(b)
    if b, s = 'yes'; else, s = 'no'; end
end


function report(stepName, err)
    fprintf(2, '  %s FAILED: %s\n', stepName, err.message);
    fprintf(2, '    identifier: %s\n', err.identifier);
    for k = 1:numel(err.stack)
        fprintf(2, '    in %s at line %d\n', err.stack(k).name, err.stack(k).line);
    end
    fprintf('\n');
end


function img = forceRgb(img, map)
%FORCERGB  What toUint8Rgb does, plus the indexed-image case it does not cover.
    if ~isempty(map)
        img = im2uint8(ind2rgb(img, map));
    end
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end
    nCh = size(img, 3);
    if nCh == 1
        img = repmat(img, [1 1 3]);
    elseif nCh == 2
        img = repmat(img(:,:,1), [1 1 3]);
    elseif nCh >= 4
        img = img(:,:,1:3);
    end
end
