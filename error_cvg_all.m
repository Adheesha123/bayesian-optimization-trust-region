clc; clear; close all;
%% ========================================================================
%  ERROR COMPARISON ACROSS ALL 7 TEST FUNCTIONS
%  ========================================================================
%  Collects final error statistics for Standard BO vs Adaptive BO.
%  Produces summary table with error reduction percentages.
%  ========================================================================

%% 1. SETTINGS (match run_comparison.m exactly)
n_trials            = 20;
N_iter              = 100;
n_init              = 5;
noise_var           = 1e-6;
length_scale_factor = 0.1;
alpha_prune         = 0.1;
beta_trust          = 0.9;
epsilon             = 1e-12;

%% 2. LOAD TEST FUNCTIONS
tests      = test_functions();
n_funcs    = length(tests);
func_names = {tests.func_name};

%% 3. STORAGE
error_standard = zeros(n_funcs, n_trials);  % Final error for each trial
error_adaptive = zeros(n_funcs, n_trials);

%% 4. MAIN LOOP
fprintf('\n========================================================================\n');
fprintf('  ERROR COLLECTION  (%d trials x %d iters x %d functions)\n', ...
        n_trials, N_iter, n_funcs);
fprintf('========================================================================\n\n');

for fi = 1:n_funcs
    tc = tests(fi);
    fprintf('[%d/%d] %s (f* = %.6f)\n', fi, n_funcs, tc.func_name, tc.f_opt);

    for trial = 1:n_trials
        seed = trial * 1000; 

        % --- Standard BO ---
        out_std = bo_gp(tc, 'EI', N_iter, n_init, noise_var, ...
                        length_scale_factor, epsilon, seed);
        error_standard(fi, trial) = abs(tc.f_opt - out_std.best_so_far(end));

        % --- Adaptive BO (selective pruning) ---
        out_adap = bo_gp_adaptiveTR_selective_pruning(tc, 'EI', N_iter, n_init, ...
                                                       noise_var, length_scale_factor, ...
                                                       alpha_prune, beta_trust, epsilon, seed);
        error_adaptive(fi, trial) = abs(tc.f_opt - out_adap.best_so_far(end));

        fprintf('  Trial %2d/%d  | Std: %.4e  | Adap: %.4e\n', ...
                trial, n_trials, error_standard(fi,trial), error_adaptive(fi,trial));
    end
    fprintf('\n');
end

%% 5. STATISTICAL ANALYSIS
mean_error_std  = mean(error_standard, 2);  % [n_funcs x 1]
mean_error_adap = mean(error_adaptive, 2);
std_error_std   = std(error_standard, 0, 2);
std_error_adap  = std(error_adaptive, 0, 2);
sem_error_std   = std_error_std / sqrt(n_trials);
sem_error_adap  = std_error_adap / sqrt(n_trials);

% Calculate error reduction percentage
error_reduction = ((mean_error_std - mean_error_adap) ./ mean_error_std) * 100;

%% 6. CONSOLE SUMMARY TABLE
fprintf('========================================================================\n');
fprintf('  ERROR PERFORMANCE SUMMARY  (mean over %d trials)\n', n_trials);
fprintf('========================================================================\n');
fprintf('  %-18s | Standard Error  | Adaptive Error  | Error Reduction\n', 'Function');
fprintf('------------------------------------------------------------------------\n');
for fi = 1:n_funcs
    fprintf('  %-18s | %10.4e     | %10.4e     | %7.1f%%\n', ...
            func_names{fi}, ...
            mean_error_std(fi), ...
            mean_error_adap(fi), ...
            error_reduction(fi));
end
fprintf('========================================================================\n');
fprintf('  Overall Average Error Reduction: %.1f%%\n', mean(error_reduction));
fprintf('========================================================================\n\n');

%% 7. DETAILED STATISTICS
fprintf('========================================================================\n');
fprintf('  DETAILED STATISTICS\n');
fprintf('========================================================================\n');
fprintf('  %-18s | Std (Mean±Std)        | Adap (Mean±Std)       | Reduction\n', 'Function');
fprintf('------------------------------------------------------------------------\n');
for fi = 1:n_funcs
    fprintf('  %-18s | %.2e±%.2e   | %.2e±%.2e   | %6.1f%%\n', ...
            func_names{fi}, ...
            mean_error_std(fi), std_error_std(fi), ...
            mean_error_adap(fi), std_error_adap(fi), ...
            error_reduction(fi));
end
fprintf('========================================================================\n\n');

%% 8. LATEX TABLE OUTPUT
fprintf('========================================================================\n');
fprintf('  LATEX TABLE FORMAT\n');
fprintf('========================================================================\n');
fprintf('\\begin{tabular}{lccc}\n');
fprintf('\\hline\n');
fprintf('\\textbf{Function} & \\textbf{Standard Error} & \\textbf{Adaptive Error} & \\textbf{Error Reduction} \\\\\n');
fprintf('\\hline\n');
for fi = 1:n_funcs
    fprintf('%s & %.2e & %.2e & %.1f\\%%%% \\\\\n', ...
            func_names{fi}, ...
            mean_error_std(fi), ...
            mean_error_adap(fi), ...
            error_reduction(fi));
end
fprintf('\\hline\n');
fprintf('\\textbf{Average} & -- & -- & \\textbf{%.1f\\%%%%} \\\\\n', mean(error_reduction));
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('========================================================================\n\n');

% %% 9. VISUALIZATION - ERROR REDUCTION BAR CHART
% figure('Color', 'w', 'Position', [100 100 1000 500]);
% hold on; grid on; box on;
% 
% % Create bar chart
% x_pos = 1:n_funcs;
% colors = zeros(n_funcs, 3);
% for fi = 1:n_funcs
%     if error_reduction(fi) > 0
%         colors(fi, :) = [0.2 0.7 0.3];  % Green for improvement
%     else
%         colors(fi, :) = [0.9 0.2 0.2];  % Red for degradation
%     end
% end
% 
% b = bar(x_pos, error_reduction, 0.7);
% b.FaceColor = 'flat';
% b.CData = colors;
% b.EdgeColor = 'k';
% b.LineWidth = 1;
% 
% % Add horizontal line at 0%
% plot([0.5, n_funcs+0.5], [0, 0], 'k--', 'LineWidth', 1.5);
% 
% % Add value labels
% for fi = 1:n_funcs
%     if error_reduction(fi) > 0
%         v_align = 'bottom';
%         y_offset = 2;
%     else
%         v_align = 'top';
%         y_offset = -2;
%     end
%     text(fi, error_reduction(fi) + y_offset, sprintf('%.1f%%', error_reduction(fi)), ...
%          'HorizontalAlignment', 'center', 'VerticalAlignment', v_align, ...
%          'FontSize', 9, 'FontWeight', 'bold');
% end
% 
% % Labels and formatting
% xticks(x_pos);
% xticklabels(func_names);
% xtickangle(20);
% ylabel('Error Reduction (%)', 'FontWeight', 'bold', 'FontSize', 12);
% xlabel('Test Function', 'FontWeight', 'bold', 'FontSize', 12);
% title(sprintf('Error Reduction: Adaptive vs Standard BO (%d trials)', n_trials), ...
%       'FontSize', 13, 'FontWeight', 'bold');
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 10);
% 
% % Add legend
% legend({'Improvement', 'Degradation'}, 'Location', 'best', 'FontSize', 10);

figure('Color', 'w', 'Position', [100 100 1200 520]);
hold on; grid on; box on;

bar_data = [mean_error_std, mean_error_adap];  % [n_funcs x 2]
b = bar(1:n_funcs, bar_data, 0.72, 'grouped');
b(1).FaceColor = [0.0 0.4 0.8];  % Blue - Standard
b(1).DisplayName = 'Standard BO';
b(2).FaceColor = [0.85 0.2 0.2]; % Red - Adaptive
b(2).DisplayName = 'Modified BO';

set(gca, 'YScale', 'log');  % LOG SCALE important since errors vary widely
xticks(1:n_funcs);
xticklabels(func_names);
xtickangle(15);
ylabel('Final Optimization Error (Log Scale)', 'FontWeight', 'bold', 'FontSize', 12);
xlabel('Test Function', 'FontWeight', 'bold', 'FontSize', 12);
title('Final Error Comparison: Standard vs Modified BO', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11);
grid on;

%% 10. SAVE RESULTS
results.func_names       = func_names;
results.error_standard   = error_standard;
results.error_adaptive   = error_adaptive;
results.mean_error_std   = mean_error_std;
results.mean_error_adap  = mean_error_adap;
results.error_reduction  = error_reduction;
results.settings.n_trials = n_trials;
results.settings.N_iter   = N_iter;

save('error_comparison_results.mat', 'results');

fprintf('✓ Results saved to: error_comparison_results.mat\n');
fprintf('✓ Error comparison complete!\n\n')



