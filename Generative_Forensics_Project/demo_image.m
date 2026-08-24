function demo_image(imgPath, outDir)
%DEMO_IMAGE  Live demonstration of the detector on a single image.
%
%   demo_image                     opens a file picker
%   demo_image('cat.jpg')          analyses that file directly
%   demo_image('cat.jpg', OUTDIR)  headless export mode for the GUI
%
%   Export mode writes OUTDIR/temp_features.csv and OUTDIR/dsp_visuals.png and
%   returns immediately - no window, no Python call. That is the handoff app.py
%   (the Streamlit GUI) uses: MATLAB measures and draws, the GUI owns the model,
%   the score and the SHAP explanation, exactly as the interactive path splits
%   the work between MATLAB and predict_image.py.
%
%   Splits the work the way the project does: MATLAB measures the image, Python
%   decides what it is. The 230 DSP features come from extractImageFeatures() -
%   the same function feature_extractor.m used to build the training set - and
%   are handed to predict_image.py, which owns the classifier.
%
%   Also opens a figure showing the signal processing behind the verdict: the
%   analysed crop, its Fourier spectrum, the finest wavelet detail band, and the
%   high-pass noise residual.
%
%   Before the first run:  python train_model.py   (produces model.joblib)

    FEATURES_CSV = 'demo_features.csv';
    PY_SCRIPT    = 'predict_image.py';

    % Export mode is checked first: it needs neither the Python script nor a
    % trained model, because the caller already owns both.
    if nargin >= 2 && ~isempty(outDir)
        exportForGui(imgPath, char(outDir));
        return;
    end

    if ~isfile(PY_SCRIPT)
        error('demo_image:noScript', 'Cannot find %s', fullfile(pwd, PY_SCRIPT));
    end
    if ~isfile('model.joblib')
        error('demo_image:noModel', ...
              'Cannot find model.joblib - run  python train_model.py  first.');
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

    %% ------------------------------------------------- measure, in MATLAB
    original = imread(imgPath);
    [features, crop, grayD] = extractImageFeatures(imgPath, 'crop');

    % One row, no label - predict_image.py expects exactly the 230 features.
    writematrix(features, FEATURES_CSV);

    [~, name, ext] = fileparts(imgPath);
    fprintf('\nMeasured %s%s -> %d features -> handing to Python\n\n', ...
            name, ext, numel(features));

    %% ------------------------------------------------- classify, in Python
    cmd = sprintf('python "%s" "%s"', PY_SCRIPT, FEATURES_CSV);
    [status, output] = system(cmd);

    if status ~= 0
        % Windows installs sometimes expose only the launcher, not `python`.
        [status, output] = system(sprintf('py "%s" "%s"', PY_SCRIPT, FEATURES_CSV));
    end

    if status ~= 0
        fprintf(2, 'Could not run Python. Output was:\n%s\n', output);
        fprintf(2, ['\nThe features were still written to %s - you can get the\n' ...
                    'verdict by running this yourself:\n    python %s %s\n'], ...
                FEATURES_CSV, PY_SCRIPT, FEATURES_CSV);
        return;
    end

    [prob, threshold, verdict] = parseVerdict(output);
    disp(stripMachineLines(output));

    if isnan(prob)
        fprintf(2, 'Could not read a score back from Python; skipping the figure.\n');
        return;
    end

    [h, w, ~] = size(original);
    if h < 256 || w < 256
        fprintf(2, ['  UNRELIABLE: image is %dx%d, smaller than the 256x256\n' ...
                    '  analysis window, so it was scaled up before measurement.\n' ...
                    '  Every training image was large enough to crop.\n'], w, h);
    end

    %% -------------------------------------------------- DSP intermediates
    spectrum = log1p(abs(fftshift(fft2(grayD))));

    [C, Sz] = wavedec2(grayD, 2, 'db4');
    [~, ~, cD1] = detcoef2('all', C, Sz, 1);

    residual = grayD - imgaussfilt(grayD, 1);

    isAI = prob >= threshold;
    if isAI
        verdictC = [0.85 0.30 0.10];
    else
        verdictC = [0.05 0.42 0.72];
    end

    %% ------------------------------------------------------------ figure
    % The printed verdict above is the demonstration's real output. Plotting is
    % guarded so a graphics problem on an unfamiliar machine degrades to "no
    % picture" rather than taking the demo down mid-presentation.
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

        subplot(2,3,6);
        hold on;
        fill([0 threshold threshold 0], [0 0 1 1], [0.05 0.42 0.72], ...
             'FaceAlpha', 0.15, 'EdgeColor', 'none');
        fill([threshold 1 1 threshold], [0 0 1 1], [0.85 0.30 0.10], ...
             'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot([threshold threshold], [0 1], 'k-', 'LineWidth', 2);
        plot(prob, 0.5, 'o', 'MarkerSize', 15, 'MarkerFaceColor', verdictC, ...
             'MarkerEdgeColor', 'w', 'LineWidth', 2);
        text(0.03, 0.15, 'REAL', 'Color', [0.05 0.42 0.72], 'FontWeight', 'bold');
        text(0.97, 0.15, 'AI', 'Color', [0.85 0.30 0.10], 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'right');
        text(prob, 0.72, sprintf('%.3f', prob), 'Color', verdictC, ...
             'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        xlim([0 1]); ylim([0 1]);
        set(gca, 'YTick', [], 'XTick', [0 0.5 1], 'Box', 'off');
        title({verdict, sprintf('score %.3f  vs  threshold %.3f', prob, threshold)}, ...
              'FontSize', 13, 'Color', verdictC);
        hold off;

    catch figErr
        fprintf(2, '  (figure could not be drawn: %s)\n', figErr.message);
        fprintf(2, '  The verdict above is still valid.\n');
    end
end


%% ================================================================
%  Local functions
%  ================================================================

function [prob, threshold, verdict] = parseVerdict(output)
%PARSEVERDICT  Read the machine-readable header predict_image.py prints first.

    prob      = NaN;
    threshold = NaN;
    verdict   = '';

    lines = strsplit(output, {sprintf('\n'), sprintf('\r')});
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if startsWith(line, 'SCORE ')
            prob = str2double(line(7:end));
        elseif startsWith(line, 'THRESHOLD ')
            threshold = str2double(line(11:end));
        elseif startsWith(line, 'VERDICT ')
            verdict = strtrim(line(9:end));
        end
    end
end


function txt = stripMachineLines(output)
%STRIPMACHINELINES  Drop the parsed header so only the human report is shown.

    lines = strsplit(output, sprintf('\n'));
    keep  = true(size(lines));
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        keep(i) = ~(startsWith(line, 'SCORE ') || ...
                    startsWith(line, 'THRESHOLD ') || ...
                    startsWith(line, 'VERDICT '));
    end
    txt = strjoin(lines(keep), sprintf('\n'));
end


function exportForGui(imgPath, outDir)
%EXPORTFORGUI  Headless measurement and figure export for the Streamlit GUI.
%
%   Writes exactly two artefacts into OUTDIR and prints one machine-readable
%   line for each, so the caller never has to infer from a filesystem race
%   whether a stage actually succeeded:
%
%       DIMS <width> <height>
%       CSV <path>            the 230 features, one row, no header
%       VISUALS <path>        the four-panel PNG, or the word  none
%       DONE
%
%   The CSV is written before the figure is drawn, and the drawing is guarded.
%   A machine with no display, no OpenGL or an unfamiliar MATLAB then costs the
%   user a picture rather than a verdict - the GUI simply omits the visual panel.

    if nargin < 1 || isempty(imgPath)
        error('demo_image:noImage', 'Export mode needs an image path.');
    end
    imgPath = char(imgPath);
    if ~isfile(imgPath)
        error('demo_image:missingImage', 'Cannot find %s', imgPath);
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    original = imread(imgPath);
    [h, w, ~] = size(original);

    % Same function the training set was built with, so the numbers the GUI
    % scores can never drift from the numbers the model learned.
    [features, crop, grayD] = extractImageFeatures(imgPath, 'crop');

    csvPath = fullfile(outDir, 'temp_features.csv');
    writematrix(features, csvPath);

    fprintf('DIMS %d %d\n', w, h);
    fprintf('CSV %s\n', csvPath);

    pngPath = fullfile(outDir, 'dsp_visuals.png');
    try
        exportPanels(crop, grayD, pngPath);
        fprintf('VISUALS %s\n', pngPath);
    catch figErr
        fprintf('VISUALS none\n');
        fprintf(2, 'figure export failed: %s\n', figErr.message);
    end

    fprintf('DONE\n');
end


function exportPanels(crop, grayD, pngPath)
%EXPORTPANELS  The four-panel PNG: what the detector actually looked at.
%
%   Crop, spectrum, finest wavelet detail and noise residual - the same
%   intermediates the interactive figure shows, minus the score gauge, which
%   the GUI draws itself from the calibrated probability.

    spectrum = log1p(abs(fftshift(fft2(grayD))));

    [C, Sz] = wavedec2(grayD, 2, 'db4');
    [~, ~, cD1] = detcoef2('all', C, Sz, 1);

    residual = grayD - imgaussfilt(grayD, 1);

    fig = figure('Visible', 'off', 'Color', 'w', ...
                 'InvertHardcopy', 'off', 'Position', [0 0 1600 430]);
    closeFig = onCleanup(@() close(fig));   % also fires if an axis call throws

    ax = gobjects(1, 4);

    ax(1) = subplot(1, 4, 1);
    imshow(crop, 'Parent', ax(1));

    ax(2) = subplot(1, 4, 2);
    imagesc(ax(2), spectrum);
    colormap(ax(2), hot(256));

    ax(3) = subplot(1, 4, 3);
    imagesc(ax(3), abs(cD1));
    colormap(ax(3), hot(256));

    lim = max(abs(residual(:)));
    if lim == 0
        lim = 1;               % a perfectly flat residual would give an empty range
    end
    ax(4) = subplot(1, 4, 4);
    imagesc(ax(4), residual, [-lim lim]);
    colormap(ax(4), gray(256));

    titles = {'Analysed crop  256x256', ...
              'Fourier spectrum  (log)', ...
              'Wavelet cD1  (finest detail)', ...
              'High-pass noise residual'};

    for k = 1:4
        axis(ax(k), 'image');
        axis(ax(k), 'off');
        title(ax(k), titles{k}, 'FontSize', 12, 'FontWeight', 'normal', ...
              'Color', [0.16 0.17 0.19]);
    end

    if exist('exportgraphics', 'file')          % R2020a and later
        exportgraphics(fig, pngPath, 'Resolution', 150, 'BackgroundColor', 'white');
    else
        print(fig, pngPath, '-dpng', '-r150');
    end
end
