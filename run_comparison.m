clc; clear; close all;
%% ========================================================================
%  BAYESIAN OPTIMIZATION: 2-METHOD COMPARISON
%  ========================================================================
%
%  This script compares:
%    1. Standard BO (Blue)
%    2. Adaptive Trust Region with Uncertainty Exploration (Red)
%
%  Tracks and visualizes uncertainty exploration samples.
%
%  ========================================================================

%% 1. EXPERIMENT SETUP

% Select the function to test
tests = test_functions();
test_case = tests(4);  % 1=Schaffer, 2=Ackley, 3=Rastrigin, 4=Branin, 5=Beale, 6=Goldstein, 7=Sphere

fprintf('\n========================================================================\n');
fprintf('  COMPARING 2 BAYESIAN OPTIMIZATION ALGORITHMS\n');
fprintf('========================================================================\n');
fprintf('  Function: %s\n', test_case.func_name);
fprintf('  Optimum:  f* = %.6f\n', test_case.f_opt);
fprintf('========================================================================\n\n');

% Experiment parameters
n_trials = 20;       % Number of independent trials for statistics
N_iter = 100;        % Iterations per trial
n_init = 5;          % Initial random samples

% Shared algorithm parameters
noise_var = 1e-6;            % Observation noise variance
length_scale_factor = 0.1;   % Length scale factor
alpha_prune = 0.1;           % Pruning radius factor
beta_trust = 0.9;            % Trust region radius factor
epsilon = 1e-12;             % Convergence threshold

% Storage allocation for both methods
hist_standard_err = zeros(n_trials, N_iter);
hist_uncertainty_err = zeros(n_trials, N_iter);

time_standard = zeros(n_trials, 1);
time_uncertainty = zeros(n_trials, 1);

%% 2. MULTI-TRIAL TOURNAMENT

fprintf('Running %d trials per method...\n', n_trials);
fprintf('------------------------------------------------------------------------\n');

for trial = 1:n_trials
    fprintf('Trial %2d/%d: ', trial, n_trials);
    seed = trial * 1000;
    
    % --- METHOD 1: Standard BO ---
    fprintf('[Standard] ');
    tic;
    out_standard = bo_gp(test_case, 'EI', N_iter, n_init, noise_var, ...
                         length_scale_factor, epsilon, seed);
    time_standard(trial) = toc;
    hist_standard_err(trial, :) = out_standard.err_history;
    
    % --- METHOD 2: Adaptive BO with Uncertainty Exploration ---
    fprintf('[Uncertainty] ');
    tic;
    out_uncertainty = bo_gp_adaptiveTR_selective_pruning(test_case, 'EI', N_iter, n_init, ...
                                                       noise_var, length_scale_factor, ...
                                                       alpha_prune, beta_trust, epsilon, seed);
    time_uncertainty(trial) = toc;
    hist_uncertainty_err(trial, :) = out_uncertainty.err_history;
    
    fprintf('✓\n');
end

fprintf('------------------------------------------------------------------------\n');
fprintf('All trials completed!\n\n');

%% 3. STATISTICAL ANALYSIS

% Compute mean and standard error (SEM)
mean_err_standard = mean(hist_standard_err, 1);
sem_err_standard = std(hist_standard_err, 0, 1) / sqrt(n_trials);

mean_err_uncertainty = mean(hist_uncertainty_err, 1);
sem_err_uncertainty = std(hist_uncertainty_err, 0, 1) / sqrt(n_trials);

% Average runtime
avg_time_standard = mean(time_standard);
avg_time_uncertainty = mean(time_uncertainty);

%% 4. VISUALIZATION

% =========================================================================
% PLOT 1: CONVERGENCE COMPARISON WITH CONFIDENCE BANDS
% =========================================================================
figure('Name', 'Convergence Comparison - 2 Methods', 'Color', 'w', ...
       'Position', [100 100 1000 600]);
hold on; grid on; box on;

iterations = 1:N_iter;

% --- Standard BO (Blue) ---
fill([iterations, fliplr(iterations)], ...
     [mean_err_standard + sem_err_standard, fliplr(mean_err_standard - sem_err_standard)], ...
     [0.0 0.4 0.8], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
h1 = plot(iterations, mean_err_standard, '-', 'Color', [0.0 0.4 0.8], ...
          'LineWidth', 2.5, 'DisplayName', 'Standard BO');

% --- Adaptive with Uncertainty (Red) ---
fill([iterations, fliplr(iterations)], ...
     [mean_err_uncertainty + sem_err_uncertainty, fliplr(mean_err_uncertainty - sem_err_uncertainty)], ...
     [0.85 0.2 0.2], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
h4 = plot(iterations, mean_err_uncertainty, '-', 'Color', [0.85 0.2 0.2], ...
          'LineWidth', 2.5, 'DisplayName', 'Adaptive (Uncertainty)');

set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Iteration', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Log Error |f^* - f_{best}|', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('Convergence Comparison (%d trials): %s', n_trials, test_case.func_name), ...
      'FontSize', 13, 'FontWeight', 'bold');
legend([h1, h4], 'Location', 'northeast', 'FontSize', 11);
xlim([1 N_iter]);
grid on;
set(gca, 'GridAlpha', 0.3);

% =========================================================================
% PLOT 2: EFFICIENCY SCATTER (Final Error vs Runtime)
% =========================================================================
figure('Name', 'Efficiency Trade-off', 'Color', 'w', 'Position', [1150 100 700 600]);
hold on; grid on; box on;

% Individual trials (semi-transparent)
scatter(time_standard, hist_standard_err(:, end), 60, [0.0 0.4 0.8], 'o', ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
scatter(time_uncertainty, hist_uncertainty_err(:, end), 60, [0.85 0.2 0.2], '^', ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');

% Mean values (bold markers)
h1 = scatter(avg_time_standard, mean_err_standard(end), 200, [0.0 0.4 0.8], 'o', ...
             'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'Standard BO');
h4 = scatter(avg_time_uncertainty, mean_err_uncertainty(end), 200, [0.85 0.2 0.2], '^', ...
             'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'Adaptive (Uncertainty)');

set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Total Runtime (seconds)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Final Error (Log Scale)', 'FontWeight', 'bold', 'FontSize', 12);
title('Efficiency Trade-off: Lower-Left is Better', 'FontSize', 13, 'FontWeight', 'bold');
legend([h1, h4], 'Location', 'best', 'FontSize', 11);
grid on;
set(gca, 'GridAlpha', 0.3);

% =========================================================================
% PLOT 3: RUNTIME BAR CHART
% =========================================================================
figure('Name', 'Runtime Comparison', 'Color', 'w', 'Position', [100 750 800 400]);

methods_names = {'Standard', 'Adaptive\n(Uncertainty)'};
means = [avg_time_standard, avg_time_uncertainty];
x_loc = 1:2;

b = bar(x_loc, means, 0.6);
b.FaceColor = 'flat';
b.EdgeColor = 'k';
b.LineWidth = 1.2;

% Set colors
b.CData(1, :) = [0.0 0.4 0.8];   % Blue
b.CData(2, :) = [0.85 0.2 0.2];   % Red

xticks(x_loc);
xticklabels(methods_names);
ylabel('Average Runtime (seconds)', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('Runtime Comparison: %s', test_case.func_name), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on; box off;
set(gca, 'FontSize', 11);

% Add value labels on top of bars
for i = 1:length(means)
    text(x_loc(i), means(i), sprintf('%.2fs', means(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 10);
end

% =========================================================================
% PLOT 4: PERFORMANCE SUMMARY TABLE
% =========================================================================
figure('Name', 'Performance Summary', 'Color', 'w', 'Position', [950 750 700 400]);
axis off;

% Calculate statistics
final_errors = [mean_err_standard(end), mean_err_uncertainty(end)];
[sorted_errors, rank_idx] = sort(final_errors);
ranks = zeros(1, 2);
ranks(rank_idx) = 1:2;

% Create table data
method_names_full = {'Standard BO', 'Adaptive (Uncertainty)'};
colors_cell = {[0.0 0.4 0.8], [0.85 0.2 0.2]};

% Table header
text(0.05, 0.95, '{\bf PERFORMANCE SUMMARY}', 'FontSize', 14);
text(0.05, 0.88, repmat('─', 1, 80), 'FontSize', 10);

% Column headers
text(0.05, 0.82, '{\bf Method}', 'FontSize', 11);
text(0.35, 0.82, '{\bf Final Error}', 'FontSize', 11);
text(0.55, 0.82, '{\bf Avg Time (s)}', 'FontSize', 11);
text(0.75, 0.82, '{\bf Rank}', 'FontSize', 11);
text(0.05, 0.78, repmat('─', 1, 80), 'FontSize', 10);

% Table rows
y_pos = 0.72;
for i = 1:2
    % Method name with color
    text(0.05, y_pos, method_names_full{i}, 'FontSize', 10, 'Color', colors_cell{i});
    
    % Final error
    text(0.35, y_pos, sprintf('%.4e', final_errors(i)), 'FontSize', 10);
    
    % Average time
    text(0.55, y_pos, sprintf('%.2f', means(i)), 'FontSize', 10);
    
    % Rank
    rank_str = sprintf('#%d', ranks(i));
    if ranks(i) == 1
        text(0.75, y_pos, rank_str, 'FontSize', 10, 'Color', [0 0.6 0], 'FontWeight', 'bold');
    else
        text(0.75, y_pos, rank_str, 'FontSize', 10);
    end
    
    y_pos = y_pos - 0.10;
end

% Winner announcement
text(0.05, 0.50, repmat('─', 1, 80), 'FontSize', 10);
text(0.05, 0.43, '{\bf WINNER:}', 'FontSize', 12);
winner_name = method_names_full{rank_idx(1)};
winner_color = colors_cell{rank_idx(1)};
text(0.25, 0.43, winner_name, 'FontSize', 12, 'Color', winner_color, 'FontWeight', 'bold');

% Statistics note
text(0.05, 0.05, sprintf('Based on %d independent trials', n_trials), ...
     'FontSize', 9, 'FontAngle', 'italic', 'Color', [0.5 0.5 0.5]);

%% 5. CONSOLE REPORT

fprintf('\n========================================================================\n');
fprintf('  FINAL RESULTS SUMMARY\n');
fprintf('========================================================================\n');
fprintf('  Method                     | Final Error (Mean) | Avg Time (s) | Rank\n');
fprintf('------------------------------------------------------------------------\n');
for i = 1:2
    fprintf('  %-25s | %.6e       | %7.2f      |  #%d\n', ...
            method_names_full{i}, final_errors(i), means(i), ranks(i));
end
fprintf('========================================================================\n\n');

% Detailed statistics
fprintf('Detailed Statistics:\n');
fprintf('------------------------------------------------------------------------\n');

% For Standard BO
fprintf('%s:\n', method_names_full{1});
fprintf('  Error - Mean: %.4e, Std: %.4e\n', ...
        final_errors(1), std(hist_standard_err(:,end)));
fprintf('  Time  - Mean: %.2fs, Std: %.2fs\n\n', ...
        means(1), std(time_standard));

% For Adaptive (Uncertainty)
fprintf('%s:\n', method_names_full{2});
fprintf('  Error - Mean: %.4e, Std: %.4e\n', ...
        final_errors(2), std(hist_uncertainty_err(:,end)));
fprintf('  Time  - Mean: %.2fs, Std: %.2fs\n\n', ...
        means(2), std(time_uncertainty));

fprintf('========================================================================\n');
fprintf('  ★ WINNER: %s\n', winner_name);
fprintf('========================================================================\n\n');

fprintf('\n✓ Comparison complete! Check the generated figures.\n\n');