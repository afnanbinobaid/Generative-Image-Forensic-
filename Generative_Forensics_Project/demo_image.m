function demo_image(imgPath)
%DEMO_IMAGE  Live demonstration of the detector on a single image.
%
%   demo_image             opens a file picker
%   demo_image('cat.jpg')  analyses that file directly
%
%   Opens one figure showing the verdict and, beside it, the signal processing
%   the verdict is actually based on: the analysed crop, its Fourier spectrum,
%   the finest wavelet detail band, and the high-pass noise residual. Those
%   panels are the point of the demonstration - they show the decision comes
%   from measurable DSP evidence rather than an opaque learned texture.
%
%   Features come from extractImageFeatures(), the same function
%   feature_extractor.m used to build the training set, so what is measured
%   here cannot drift from what the model was trained on.
%
%   Run train_model.m once first to produce model.mat.

    MODEL_PATH = 'model.mat';

    if ~isfile(MODEL_PATH)
        error('demo_image:noModel', ...
              ['Cannot find %s - run train_model.m first.'], ...
              fullfile(pwd, MODEL_PATH));
    end

    if nargin < 1 || isempty(imgPath)
        [f, p] = uigetfile({'*.jpg;*.jpeg;*.png', 'Images (*.jpg, *.jpeg, *.png)'}, ...
                           'Pick an image to analyse');
        if isequal(f, 0)
            fprintf('Cancelled.\n');
            return;
        end
        imgPath = fullfile(p, f);
    end

    S = load(MODEL_PATH);

    %% ---------------------------------------------------------- measure
    original = imread(imgPath);
    [features, crop, grayD] = extractImageFeatures(imgPath, 'crop');

    [~, scores] = predict(S.model, features);
    prob = scores(2);

    isAI    = prob >= S.threshold;
    if isAI
        verdict   = 'AI-GENERATED';
        verdictC  = [0.85 0.30 0.10];
        headroom  = 1 - S.threshold;
    else
        verdict   = 'REAL PHOTOGRAPH';
        verdictC  = [0.05 0.42 0.72];
        headroom  = S.threshold;
    end
    margin   = abs(prob - S.threshold) / max(headroom, 1e-9);
    if margin > 0.6
        strength = 'strong';
    elseif margin > 0.25
        strength = 'moderate';
    else
        strength = 'weak';
    end

    %% -------------------------------------------------- DSP intermediates
    spectrum = log1p(abs(fftshift(fft2(grayD))));

    [C, Sz] = wavedec2(grayD, 2, 'db4');
    [~, ~, cD1] = detcoef2('all', C, Sz, 1);

    residual = grayD - imgaussfilt(grayD, 1);

    %% ------------------------------------------------------------ report
    [~, name, ext] = fileparts(imgPath);
    fprintf('\n=============================================\n');
    fprintf('  %s%s\n', name, ext);
    fprintf('  Verdict   : %s\n', verdict);
    fprintf('  Score     : %.4f   (threshold %.4f)\n', prob, S.threshold);
    fprintf('  Separation: %s\n', strength);
    fprintf('=============================================\n');

    [h, w, ~] = size(original);
    if h < 256 || w < 256
        fprintf(2, ['  UNRELIABLE: image is %dx%d, smaller than the 256x256\n' ...
                    '  analysis window, so it was scaled up before measurement.\n' ...
                    '  Every training image was large enough to crop, so this\n' ...
                    '  result is outside what the model has seen.\n'], w, h);
    end

    fprintf('\n  %-34s %10s %10s %10s\n', 'measurement', 'this', 'real', 'AI');
    fprintf('  %s\n', repmat('-', 1, 66));
    for i = 1:6
        c = S.topFeatures(i);
        fprintf('  %-34s %10.4g %10.4g %10.4g\n', ...
                describeFeature(c), features(c), S.meanReal(c), S.meanAI(c));
    end
    fprintf('\n');

    %% ------------------------------------------------------------ figure
    % The console report above is the demonstration's core output. Plotting is
    % wrapped so that a graphics problem on an unfamiliar machine degrades to
    % "no picture" rather than taking the whole demo down mid-presentation.
    try
        figure('Name', sprintf('Detector - %s%s', name, ext), ...
               'Color', 'w', 'Position', [80 80 1180 700]);

        subplot(2,3,1);
        imshow(original);
        title(sprintf('Input  (%d x %d)', w, h), 'FontSize', 11);

        subplot(2,3,2);
        imshow(crop);
        title({'Analysis crop  256x256', 'native scale, no resampling'}, ...
              'FontSize', 11);

        subplot(2,3,3);
        imagesc(spectrum);
        axis image; axis off; colormap(gca, hot(256));
        title('Fourier spectrum  (log)', 'FontSize', 11);

        subplot(2,3,4);
        imagesc(abs(cD1));
        axis image; axis off; colormap(gca, hot(256));
        title('Wavelet cD1  (finest detail)', 'FontSize', 11);

        subplot(2,3,5);
        lim = max(abs(residual(:)));
        if lim == 0
            lim = 1;
        end
        imagesc(residual, [-lim lim]);
        axis image; axis off; colormap(gca, gray(256));
        title('High-pass noise residual', 'FontSize', 11);

        % Verdict panel: the score placed on a bar with the threshold marked.
        subplot(2,3,6);
        hold on;
        fill([0 S.threshold S.threshold 0], [0 0 1 1], [0.05 0.42 0.72], ...
             'FaceAlpha', 0.15, 'EdgeColor', 'none');
        fill([S.threshold 1 1 S.threshold], [0 0 1 1], [0.85 0.30 0.10], ...
             'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot([S.threshold S.threshold], [0 1], 'k-', 'LineWidth', 2);
        plot(prob, 0.5, 'o', 'MarkerSize', 15, 'MarkerFaceColor', verdictC, ...
             'MarkerEdgeColor', 'w', 'LineWidth', 2);
        text(0.03, 0.15, 'REAL', 'Color', [0.05 0.42 0.72], 'FontWeight', 'bold');
        text(0.97, 0.15, 'AI', 'Color', [0.85 0.30 0.10], 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'right');
        text(prob, 0.72, sprintf('%.3f', prob), 'Color', verdictC, ...
             'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        xlim([0 1]); ylim([0 1]);
        set(gca, 'YTick', [], 'XTick', [0 0.5 1], 'Box', 'off');
        title({verdict, sprintf('score %.3f vs threshold %.3f  (%s)', ...
               prob, S.threshold, strength)}, ...
              'FontSize', 13, 'Color', verdictC);
        hold off;

    catch figErr
        fprintf(2, '  (figure could not be drawn: %s)\n', figErr.message);
        fprintf(2, '  The verdict above is still valid.\n');
    end
end
