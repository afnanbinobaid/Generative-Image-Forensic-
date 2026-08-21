function label = describeFeature(col)
%DESCRIBEFEATURE  What a feature column actually measures.
%   describeFeature(95) returns 'wavelet cD1 std (red)'.
%
%   COL is the 1-based column number, matching the CSV written by
%   feature_extractor.m and the column numbers printed by the analysis scripts.

    channels  = {'red', 'green', 'blue'};
    baseStats = {'mean', 'std', 'variance', 'energy', 'entropy', ...
                 'skewness', 'kurtosis'};
    glcmStats = {'GLCM contrast', 'GLCM correlation', 'GLCM energy', ...
                 'GLCM homogeneity'};
    edgeStats = {'mean Gx', 'var Gx', 'mean Gy', 'var Gy', 'mean |G|', 'var |G|'};
    subbands  = {'cA2', 'cH2', 'cV2', 'cD2', 'cH1', 'cV1', 'cD1'};
    residuals = {'Laplacian residual', 'median residual', 'Gaussian residual'};
    resStats  = {'mean', 'std', 'kurtosis'};
    pairs     = {'R-G', 'R-B', 'G-B'};

    if col < 1 || col > 230
        error('describeFeature:range', 'Column must be 1..230, got %d', col);
    end

    c0 = col - 1;                       % 0-based makes the arithmetic simpler

    if c0 < 51                                          % Block A
        ch = floor(c0 / 17);
        k  = mod(c0, 17);
        if k < 7
            what = baseStats{k + 1};
        elseif k < 11
            what = glcmStats{k - 7 + 1};
        else
            what = edgeStats{k - 11 + 1};
        end
        label = sprintf('%s (%s)', what, channels{ch + 1});

    elseif c0 < 198                                     % Block B
        ch   = floor((c0 - 51) / 49);
        k    = mod(c0 - 51, 49);
        sub  = floor(k / 7);
        stat = mod(k, 7);
        label = sprintf('wavelet %s %s (%s)', subbands{sub + 1}, ...
                        baseStats{stat + 1}, channels{ch + 1});

    elseif c0 < 218                                     % Block C
        label = sprintf('FFT ring %d of 20', c0 - 197);

    elseif c0 < 227                                     % Block D
        f = floor((c0 - 218) / 3);
        k = mod(c0 - 218, 3);
        label = sprintf('%s %s', residuals{f + 1}, resStats{k + 1});

    else                                                % Block E
        label = sprintf('channel correlation %s', pairs{c0 - 227 + 1});
    end
end
