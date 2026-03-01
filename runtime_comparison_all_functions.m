clc; clear; close all;
%% ========================================================================
%  RUNTIME COMPARISON ACROSS ALL 7 TEST FUNCTIONS
%  ========================================================================
%  Identical settings to run_comparison.m.
%  Produces one grouped bar chart summarising average runtime per function.
%  ========================================================================

%% 1. SETTINGS  (must match run_comparison.m exactly)
n_trials            = 20;
N_iter              = 100;
n_init              = 5;
noise_var           = 1e-6;
length_scale_factor = 0.1;
alpha_prune         = 0.1;
beta_trust          = 0.9;
epsilon             = 1e-12;

%% 2. LOAD TEST FUNCTIONS
tests   = test_functions();
n_funcs = length(tests);
func_names = {tests.func_name};

%% 3. STORAGE
time_standard   = zeros(n_funcs, n_trials);
time_adaptive   = zeros(n_funcs, n_trials);

%% 4. MAIN LOOP
fprintf('\n========================================================================\n');
fprintf('  RUNTIME COLLECTION  (%d trials x %d iters x %d functions)\n', ...
        n_trials, N_iter, n_funcs);
fprintf('========================================================================\n\n');

for fi = 1:n_funcs
    tc = tests(fi);
    fprintf('[%d/%d] %s\n', fi, n_funcs, tc.func_name);

    for trial = 1:n_trials
        seed = trial * 1000; 

        % Standard BO
        tic;
        bo_gp(tc, 'EI', N_iter, n_init, noise_var, ...
              length_scale_factor, epsilon, seed);
        time_standard(fi, trial) = toc;

        % Adaptive BO (selective pruning)
        tic;
        bo_gp_adaptiveTR_selective_pruning(tc, 'EI', N_iter, n_init, noise_var, ...
              length_scale_factor, alpha_prune, beta_trust, epsilon, seed);
        time_adaptive(fi, trial) = toc;

        fprintf('  trial %2d/%d  | std=%.2fs  adap=%.2fs\n', ...
                trial, n_trials, time_standard(fi,trial), time_adaptive(fi,trial));
    end
    fprintf('\n');
end

%% 5. AGGREGATE
mean_std  = mean(time_standard, 2);   % [n_funcs x 1]
mean_adap = mean(time_adaptive,  2);
sem_std   = std(time_standard,  0, 2) / sqrt(n_trials);
sem_adap  = std(time_adaptive,  0, 2) / sqrt(n_trials);

%% 6. GROUPED BAR CHART
c_std  = [0.00 0.40 0.80];   % Blue  (Standard BO)
c_adap = [0.85 0.20 0.20];   % Red   (Adaptive BO)

figure('Color', 'w', 'Position', [80 80 1200 520]);
hold on; grid on; box on;

bar_data   = [mean_std, mean_adap];          % [n_funcs x 2]
b = bar(1:n_funcs, bar_data, 0.72, 'grouped');
b(1).FaceColor    = c_std;
b(1).EdgeColor    = 'none';
b(1).DisplayName  = 'Standard BO';
b(2).FaceColor    = c_adap;
b(2).EdgeColor    = 'none';
b(2).DisplayName  = 'Modified BO';

% Error bars (SEM)
nbars      = 2;
groupwidth = min(0.8, nbars/(nbars + 1.5));
for bi = 1:nbars
    x_err = (1:n_funcs) - groupwidth/2 + (2*bi-1)*groupwidth/(2*nbars);
    err_vals = [sem_std, sem_adap];
    errorbar(x_err, bar_data(:,bi), err_vals(:,bi), ...
             'k', 'LineStyle', 'none', 'LineWidth', 1.4, ...
             'CapSize', 6, 'HandleVisibility', 'off');
end

% Value labels on top of each bar
for bi = 1:nbars
    x_pos = (1:n_funcs) - groupwidth/2 + (2*bi-1)*groupwidth/(2*nbars);
    for fi = 1:n_funcs
        val = bar_data(fi, bi);
        text(x_pos(fi), val + 0.005*max(bar_data(:)), ...
             sprintf('%.2fs', val), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment',   'bottom', ...
             'FontSize', 7.5, 'FontWeight', 'bold', ...
             'Color', [0.15 0.15 0.15]);
    end
end

% Axes & labels
xticks(1:n_funcs);
xticklabels(func_names);
xtickangle(15);
ylabel('Average Runtime (seconds)', 'FontWeight', 'bold', 'FontSize', 12);
xlabel('Test Function',             'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('Runtime Comparison: Standard BO vs Adaptive BO   (%d trials, %d iterations)', ...
              n_trials, N_iter), 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11, 'Box', 'on');
set(gca, 'GridAlpha', 0.25, 'FontSize', 10.5, 'GridLineStyle', '--');
ylim([0, max(bar_data(:)) * 1.28]);

%% 7. CONSOLE SUMMARY
fprintf('\n========================================================================\n');
fprintf('  RUNTIME SUMMARY  (mean ± SEM over %d trials)\n', n_trials);
fprintf('========================================================================\n');
fprintf('  %-18s | Standard (s)    | Adaptive (s)    | Overhead\n', 'Function');
fprintf('------------------------------------------------------------------------\n');
for fi = 1:n_funcs
    overhead = (mean_adap(fi)/mean_std(fi) - 1)*100;
    fprintf('  %-18s | %5.2f ± %4.2f    | %5.2f ± %4.2f    | %+.1f%%\n', ...
            func_names{fi}, ...
            mean_std(fi),  sem_std(fi), ...
            mean_adap(fi), sem_adap(fi), ...
            overhead);
end
fprintf('========================================================================\n');
fprintf('Overall avg — Standard: %.2fs  |  Adaptive: %.2fs\n', ...
        mean(mean_std), mean(mean_adap));
fprintf('========================================================================\n\n');
fprintf('✓ Done!\n\n');