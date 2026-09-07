function ok = normalize_image(srcPath, dstPath, cropSide, targetSide, quality)
%NORMALIZE_IMAGE  One image through the container normalisation, written as JPEG.
%
%   ok = normalize_image('in.jpg', 'out.jpg')
%   ok = normalize_image(src, dst, 448, 320, 85)
%
%   Returns true if the normalised image was written, false if the source was
%   smaller than cropSide - which is not an error, it is the model's operating
%   range. Below that floor the image cannot be measured the way the training
%   set was, and upscaling it would low-pass filter it into looking generated.
%
%   THIS IS THE ONE DEFINITION OF THE TREATMENT.
%
%   normalize_folder.m calls it for every image in a dataset, and demo_image.m
%   calls it for a single upload before measuring. That matters more than it
%   looks: a model trained on normalised images and shown a raw one is being
%   asked about a kind of picture it has never seen. A 2000px photograph
%   cropped at native scale carries far more fine detail than a 320px training
%   image does, and this detector reads fine detail. Two definitions of the
%   treatment, in two files, is exactly the drift that would put the live demo
%   and the training set back out of step.
%
%   The four steps, in this order and for every image alike:
%
%     1. centre-crop to cropSide at NATIVE SCALE, origin snapped to a multiple
%        of 8 so the window stays aligned with the JPEG DCT block grid.
%        Nothing is resampled here - this only fixes the pixel dimensions.
%     2. resample to targetSide. Because step 1 made every image the same
%        size, this is one factor for everything, which is what destroys the
%        prior 8x8 quantisation signature without manufacturing a new
%        class-correlated one.
%     3. decimate chroma explicitly - RGB to YCbCr, box-downsample Cb and Cr
%        by two, bilinearly back up, to RGB. Done in pixels because MATLAB's
%        imwrite picks subsampling from the quality value internally and does
%        not expose it.
%     4. write JPEG at one fixed quality. The encode is part of the treatment,
%        not an afterthought - the training images all carry its quantisation,
%        so a demo image must carry it too.
%
%   See normalize_dataset.m for the measured sweep behind the 448/320 choice.

    if nargin < 3 || isempty(cropSide),   cropSide   = 448; end
    if nargin < 4 || isempty(targetSide), targetSide = 320; end
    if nargin < 5 || isempty(quality),    quality    = 85;  end

    img = toUint8Rgb(imread(srcPath));

    [h, w, ~] = size(img);
    if h < cropSide || w < cropSide
        ok = false;                  % below the operating range, not an error
        return;
    end

    crop  = centreCrop(img, cropSide);
    small = imresize(crop, [targetSide targetSide], 'bicubic');
    imwrite(decimateChroma(small), dstPath, 'Quality', quality);
    ok = true;
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
%CENTRECROP  A side x side centre crop at native scale.
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
