function [ok, info] = normalize_image(srcPath, dstPath, varargin)
%NORMALIZE_IMAGE  One image through the container normalisation, written as JPEG.
%
%   ok = normalize_image('in.jpg', 'out.jpg')
%   ok = normalize_image(src, dst, opts)                 opts from normalize_defaults
%   ok = normalize_image(src, dst, 448, 320, 85)         legacy positional form
%   [ok, info] = normalize_image(...)                    what the treatment did
%
%   Returns true if the normalised image was written, false if the source was
%   below the size floor - which is not an error, it is the model's operating
%   range. Below the floor the image cannot be measured the way the training
%   set was, and upscaling it would low-pass filter it into looking generated.
%
%   INFO reports srcSize, preScale (the factor step 1 applied), analysedSide
%   (how many native pixels the measured window covers) and the settings used,
%   so a caller can say what scale the verdict was formed at.
%
%   THIS IS THE ONE DEFINITION OF THE TREATMENT.
%   normalize_defaults.m is the one definition of its numbers.
%
%   normalize_folder.m calls it for every image in a dataset, and demo_image.m
%   calls it for a single upload before measuring. That matters more than it
%   looks: a model trained on normalised images and shown a raw one is being
%   asked about a kind of picture it has never seen. Two definitions of the
%   treatment, in two files, is exactly the drift that would put the live demo
%   and the training set back out of step.
%
%   THE STEP THAT WAS MISSING, AND THE FAULT IT CAUSED
%
%   Steps 2-5 below make every image the same PIXEL SIZE. They do not make it
%   the same CONTENT SCALE, and the detector reads fine detail, which is a
%   property of scale. Cropping 448 native pixels out of a 500px photograph
%   captures the whole subject; cropping 448 out of a 4000px photograph
%   captures a fingernail, magnified, with the scene's detail spread thin
%   across the pixels. So detail-per-pixel fell monotonically as the upload
%   got bigger, and the verdict slid with it - every large image read as
%   generated and every small one as real, whatever it actually was. The
%   training classes had non-overlapping native widths (real 450-500, AI
%   512-1024), so that slide was pointed almost exactly along the label.
%
%   Step 1 removes it: resample the whole frame to a fixed short side FIRST,
%   so the crop always covers the same fraction of the picture and the
%   measurement is taken at one scale for every image. Set scaleMode to
%   'native' to reproduce the old behaviour.
%
%   The five steps, in this order and for every image alike:
%
%     1. resample the whole frame so its short side is scaleSide, keeping the
%        aspect ratio. Downscale only - an image below the floor is declined,
%        never upscaled. ('native' mode skips this step.)
%     2. centre-crop to cropSide, origin snapped to a multiple of 8 so the
%        window stays aligned with the JPEG DCT block grid.
%     3. resample to targetSide. Because step 2 made every image the same
%        size, this is one factor for everything, which is what destroys the
%        prior 8x8 quantisation signature without manufacturing a new
%        class-correlated one.
%     4. decimate chroma explicitly - RGB to YCbCr, box-downsample Cb and Cr
%        by two, bilinearly back up, to RGB. Done in pixels because MATLAB's
%        imwrite picks subsampling from the quality value internally and does
%        not expose it.
%     5. write JPEG at one fixed quality. The encode is part of the treatment,
%        not an afterthought - the training images all carry its quantisation,
%        so a demo image must carry it too.
%
%   WHAT STEP 1 DOES NOT FIX. Every image now shows the same field of view,
%   but they reach it by different resample factors - a 4000px source is
%   low-passed hard, a 460px source barely at all - and that factor is still
%   partly readable in the pixels. It is a far smaller effect than the
%   magnification difference it replaces, and normalize_folder's scale ladder
%   is what spreads it across both classes so the model cannot use it. Only a
%   dataset with matched native resolutions removes it outright.
%
%   See normalize_dataset.m for the measured sweep behind the 448/320 choice.

    opts = parseOptions(varargin);

    img = toUint8Rgb(imread(srcPath));

    [h0, w0, ~] = size(img);
    info = struct('srcSize', [w0 h0], 'preScale', 1, 'analysedSide', NaN, ...
                  'opts', opts);

    shortSide = min(h0, w0);
    floorSide = sizeFloor(opts);
    if shortSide < floorSide
        ok = false;                  % below the operating range, not an error
        return;
    end

    % --- 1. one content scale for every image ------------------------------
    if strcmpi(opts.scaleMode, 'shortside')
        info.preScale = opts.scaleSide / shortSide;

        % Sized explicitly rather than by a scalar factor: a scalar leaves the
        % short side one pixel under scaleSide whenever the division rounds
        % down, and the crop below would then not fit.
        if h0 <= w0
            newSize = [opts.scaleSide, max(opts.cropSide, round(w0 * info.preScale))];
        else
            newSize = [max(opts.cropSide, round(h0 * info.preScale)), opts.scaleSide];
        end
        img = imresize(img, newSize, 'bicubic');
    end

    % How much of the ORIGINAL picture the measured window covers, in native
    % pixels. Constant across uploads in 'shortside' mode; in 'native' mode it
    % is cropSide whatever the source was, which is the fault described above.
    info.analysedSide = opts.cropSide / info.preScale;

    crop  = centreCrop(img, opts.cropSide);
    small = imresize(crop, [opts.targetSide opts.targetSide], 'bicubic');
    imwrite(decimateChroma(small), dstPath, 'Quality', opts.quality);
    ok = true;
end


function floorSide = sizeFloor(opts)
%SIZEFLOOR  Smallest short side this treatment can be applied to.
%   'shortside' declines anything that would have to be scaled UP to reach
%   scaleSide; 'native' anything smaller than the crop. Upscaling is a low-pass
%   filter and would push the image toward the AI verdict for a reason that has
%   nothing to do with how it was made.

    if strcmpi(opts.scaleMode, 'shortside')
        floorSide = opts.scaleSide;
    else
        floorSide = opts.cropSide;
    end
end


function opts = parseOptions(args)
%PARSEOPTIONS  Accept either an options struct or the legacy positional form.
%   normalize_image(src, dst, opts)
%   normalize_image(src, dst, cropSide, targetSide, quality)

    if numel(args) == 1 && isstruct(args{1})
        opts = normalize_defaults(args{1});
        return;
    end

    over = struct();
    names = {'cropSide', 'targetSide', 'quality'};
    if numel(args) > numel(names)
        error('normalize_image:tooManyArgs', ...
              ['Too many arguments. Use normalize_image(src, dst, OPTS) with a ' ...
               'struct\nfrom normalize_defaults, or the legacy ' ...
               '(src, dst, cropSide, targetSide, quality).']);
    end
    for i = 1:numel(args)
        if ~isempty(args{i})
            over.(names{i}) = args{i};
        end
    end
    opts = normalize_defaults(over);
end



%% ================================================================
%  Local functions
%  ================================================================

function img = toUint8Rgb(img)
%TOUINT8RGB  Force any imread output into 3-channel uint8 RGB.
%   Matches extractImageFeatures.m, so the normalised image is built on the
%   same convention the features are measured with.

    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end

    nCh = size(img, 3);
    if nCh == 1
        img = repmat(img, [1 1 3]);          % grayscale -> RGB
    elseif nCh == 2
        img = repmat(img(:,:,1), [1 1 3]);   % gray + alpha -> drop alpha
    elseif nCh >= 4
        img = img(:,:,1:3);                  % RGBA / CMYK-ish -> drop extras
    end
end


function out = centreCrop(img, side)
%CENTRECROP  A side x side centre crop of whatever it is handed.
%   The origin snaps to a multiple of 8 so the crop stays aligned with the
%   JPEG DCT block grid, exactly as extractImageFeatures.m does.

    [h, w, ~] = size(img);

    r0 = floor((h - side) / 2);
    c0 = floor((w - side) / 2);
    r0 = r0 - mod(r0, 8) + 1;
    c0 = c0 - mod(c0, 8) + 1;

    out = img(r0:r0+side-1, c0:c0+side-1, :);
end


function img = decimateChroma(img)
%DECIMATECHROMA  Halve the chroma resolution, the way 4:2:0 does.
%
%   Applied to an image that was already 4:2:0 it changes very little - the
%   detail it removes is already gone - which is what makes it safe to run on
%   everything rather than only on what needs it.

    ycc = rgb2ycbcr(img);
    y   = ycc(:,:,1);
    cb  = ycc(:,:,2);
    cr  = ycc(:,:,3);

    full = size(cb);
    cb = imresize(imresize(cb, 0.5, 'box'), full, 'bilinear');
    cr = imresize(imresize(cr, 0.5, 'box'), full, 'bilinear');

    img = ycbcr2rgb(cat(3, y, cb, cr));
end
