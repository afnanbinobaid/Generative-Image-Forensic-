function opts = normalize_defaults(overrides)
%NORMALIZE_DEFAULTS  The one definition of the normalisation settings.
%
%   opts = normalize_defaults
%   opts = normalize_defaults(struct('scaleMode', 'native'))
%
%   normalize_image.m owns the treatment; this owns the NUMBERS behind it, so
%   the training set and the live demo cannot be normalised differently by a
%   value edited in one file and not the other. Change a setting here and it
%   changes for normalize_dataset.m, normalize_folder.m, demo_image.m and
%   scale_sweep.m at once.
%
%   Fields
%     scaleMode   'shortside' (default) or 'native'
%
%                 'shortside' resamples the whole frame so its SHORT SIDE is
%                 scaleSide before anything is cropped. Every image is then
%                 measured at the same content scale: the crop covers the same
%                 fraction of the picture whether the upload was 500px or
%                 5000px across.
%
%                 'native' is the original behaviour - crop cropSide pixels at
%                 native scale, no pre-scaling. It measures a 4000px photograph
%                 at a magnification no training image was ever seen at, which
%                 is what made the detector track upload resolution instead of
%                 origin. Kept only so the old pipeline can be reproduced.
%
%     scaleSide   short side to resample to in 'shortside' mode (448)
%     cropSide    centre crop, in pixels, taken after that (448)
%     targetSide  the crop is then resampled to this (320)
%     quality     JPEG quality the normalised image is written at (85)
%
%   CHANGING scaleMode INVALIDATES model.joblib. The model has only ever seen
%   images treated the way the setting here says, so after changing it:
%
%       normalize_dataset('Dataset_norm')     (MATLAB)  re-normalise
%       make_augmented                        (MATLAB)
%       feature_extractor                     (MATLAB)
%       python train_model.py
%       matlab -batch "scale_sweep('Dataset/Real_Images')"
%       python scale_sweep.py scale_sweep.csv    -> confirm the fault is gone

    opts = struct('scaleMode',  'shortside', ...
                  'scaleSide',  448, ...
                  'cropSide',   448, ...
                  'targetSide', 320, ...
                  'quality',    85);

    if nargin >= 1 && ~isempty(overrides)
        if ~isstruct(overrides)
            error('normalize_defaults:badOverrides', ...
                  'Overrides must be a struct, got %s.', class(overrides));
        end
        f = fieldnames(overrides);
        for i = 1:numel(f)
            if ~isfield(opts, f{i})
                error('normalize_defaults:unknownField', ...
                      'Unknown normalisation setting "%s".', f{i});
            end
            if ~isempty(overrides.(f{i}))
                opts.(f{i}) = overrides.(f{i});
            end
        end
    end

    validate(opts);
end


function validate(opts)
%VALIDATE  Reject a combination that would silently produce a different pipeline.

    if ~ismember(lower(opts.scaleMode), {'shortside', 'native'})
        error('normalize_defaults:badMode', ...
              'scaleMode must be ''shortside'' or ''native'', got ''%s''.', ...
              opts.scaleMode);
    end
    if mod(opts.cropSide, 8) ~= 0
        error('normalize_defaults:badCrop', ...
              'cropSide must be a multiple of 8 to keep DCT alignment (got %d).', ...
              opts.cropSide);
    end
    if opts.targetSide < 256
        error('normalize_defaults:tooSmall', ...
              ['targetSide must be at least 256 - the extractor measures a ' ...
               '256x256 window (got %d).'], opts.targetSide);
    end
    if opts.cropSide / opts.targetSide < 1.25
        error('normalize_defaults:weakResample', ...
              ['cropSide/targetSide is %.2fx. Below about 1.25x the resample no ' ...
               'longer\ndestroys the prior DCT grid and the container confound ' ...
               'survives - measured at\n1.02x (d=4.33), 1.12x (d=1.95), ' ...
               '1.27x (d=0.74), 1.40x (d=0.24).'], ...
              opts.cropSide / opts.targetSide);
    end
    if strcmpi(opts.scaleMode, 'shortside') && opts.scaleSide < opts.cropSide
        error('normalize_defaults:cropExceedsScale', ...
              ['scaleSide (%d) is below cropSide (%d): the crop would not fit ' ...
               'in the\nresampled frame.'], opts.scaleSide, opts.cropSide);
    end
end
