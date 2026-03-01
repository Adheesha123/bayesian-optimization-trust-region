function visualize_comparison()
% VISUALIZE_COMPARISON - Animated comparison of Bayesian Optimization methods
%
% =========================================================================
%                       ANIMATED COMPARISON
% =========================================================================
%
% This function creates an animated visualization comparing:
%   1. Standard BO (Blue)
%   2. Adaptive Trust Region with Uncertainty Exploration (Red)
%
% The animation shows:
%   - Spatial exploration for each method
%   - Trust regions and pruning zones (where applicable)
%   - Uncertainty exploration markers (Red stars)
%   - Real-time convergence comparison
%   - Runtime statistics
%
% =========================================================================

%% USER SETTINGS

test_id      = 2;        % 1=Schaffer, 2=Ackley, 3=Rastrigin, 4=Branin, 5=Beale, 6=Goldstein, 7=Sphere
N_iter       = 30;       % Number of BO iterations
n_init       = 5;        % Initial random samples
pause_time   = 0.3;      % Pause between frames (seconds)
save_gif     = false;    % Set true to save animated GIF

% Algorithm parameters (shared across methods)
noise_var = 1e-6;
length_scale_factor = 0.1;
alpha_prune  = 0.10;     % Pruning radius factor
beta_trust   = 0.90;     % Trust region radius factor

clc; close all;

%% LOAD TEST FUNCTION

tests = test_functions();
test = tests(test_id);

rng_seed = 1000;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     ANIMATED BAYESIAN OPTIMIZATION COMPARISON (2 METHODS)    ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Function: %-50s ║\n', test.func_name);
fprintf('║  Optimum:  f* = %-44.4f ║\n', test.f_opt);
fprintf('║  Iterations: %-46d ║\n', N_iter);
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% ========================================================================
%  RUN ALL METHODS
%  ========================================================================

methods = struct('name', {}, 'color', {}, 'result', {}, 'time', {}, ...
                 'show_tr', {}, 'show_prune', {});

% -------------------------------------------------------------------------
% METHOD 1: Standard BO (BLUE)
% -------------------------------------------------------------------------
fprintf('[1/2] Running Standard BO... ');
tic;
res = bo_gp(test, 'EI', N_iter, n_init, noise_var, length_scale_factor, 1e-12, rng_seed);
methods(1).name = 'Standard BO';
methods(1).color = [0.0 0.4 0.8];        % Blue
methods(1).result = res;
methods(1).time = toc;
methods(1).show_tr = false;              % No trust region
methods(1).show_prune = false;           % No pruning
fprintf('✓ (%.2fs)\n', methods(1).time);

% -------------------------------------------------------------------------
% METHOD 2: Adaptive BO with Uncertainty Exploration (Red)
% -------------------------------------------------------------------------
fprintf('[2/2] Running Adaptive BO (Uncertainty)... ');
tic;
res = bo_gp_adaptiveTR_selective_pruning(test, 'EI', N_iter, n_init, noise_var, ...
                                       length_scale_factor, alpha_prune, beta_trust, ...
                                       1e-12, rng_seed);
methods(2).name = 'Adaptive (Uncertainty)';
methods(2).color = [0.85 0.2 0.2];        % Red
methods(2).result = res;
methods(2).time = toc;
methods(2).show_tr = true;               % Show trust region
methods(2).show_prune = true;            % Show pruning zones
fprintf('✓ (%.2fs)\n', methods(2).time);

n_methods = length(methods);
fprintf('\n✓ All %d methods completed. Starting animation...\n\n', n_methods);

% DEBUG: Check if outside_TR_iterations field exists
fprintf('DEBUG: Checking for outside_TR_iterations field:\n');
for m = 1:n_methods
    fprintf('  %s: ', methods(m).name);
    if isfield(methods(m).result, 'outside_TR_iterations')
        n_outside = sum(methods(m).result.outside_TR_iterations);
        fprintf('✓ Found! %d/%d iterations outside TR (%.1f%%)\n', ...
                n_outside, N_iter, 100*n_outside/N_iter);
    else
        fprintf('✗ Field not found\n');
    end
end
fprintf('\n');

%% SETUP VISUALIZATION

% Domain
dom = test.domain;
if size(dom, 1) == 2
    xlims = dom(1, :);
    ylims = dom(2, :);
else
    xlims = dom;
    ylims = dom;
end

% Background surface
[X1, X2] = meshgrid(linspace(xlims(1), xlims(2), 100), ...
                    linspace(ylims(1), ylims(2), 100));
Z_true = test.f(X1, X2);

% For circles (trust regions and pruning zones)
theta = linspace(0, 2*pi, 80);

% Find minimum iterations across all methods
Kmax = N_iter;
for m = 1:n_methods
    Kmax = min(Kmax, length(methods(m).result.best_so_far));
end

% GIF setup
if save_gif
    gif_name = sprintf('comparison_2methods_%s.gif', strrep(test.func_name, ' ', '_'));
end

%% CREATE FIGURE

fig = figure('Color', 'w', 'Position', [50 50 1800 900]);

%% ========================================================================
%  ANIMATION LOOP
%  ========================================================================

for k = 1:Kmax
    clf(fig);
    n_obs = n_init + k - 1;
    
    % =====================================================================
    %  TOP ROW: SPATIAL PLOTS (2 methods) + LEGEND
    % =====================================================================
    for m = 1:n_methods
        subplot(2, n_methods + 1, m);
        hold on; box on;
        
        res = methods(m).result;
        
        % -----------------------------------------------------------------
        % Background function surface
        % -----------------------------------------------------------------
        imagesc([xlims(1) xlims(2)], [ylims(1) ylims(2)], Z_true);
        axis xy;
        colormap(gca, parula);
        h_img = findobj(gca, 'Type', 'image');
        set(h_img, 'AlphaData', 0.4);
        
        % -----------------------------------------------------------------
        % Get data for this iteration
        % -----------------------------------------------------------------
        X_obs = res.X_sample(1:n_obs, :);
        Y_obs = res.Y_sample(1:n_obs);
        [~, idx_best] = max(Y_obs);
        x_best = X_obs(idx_best, :);
        
        % -----------------------------------------------------------------
        % Trust Region (if applicable)
        % -----------------------------------------------------------------
        if methods(m).show_tr
            % Get radius (from history if available)
            if isfield(res, 'trust_radius_history') && k <= length(res.trust_radius_history)
                r = res.trust_radius_history(k);
            elseif isfield(res, 'r_tau_history') && k <= length(res.r_tau_history)
                r = res.r_tau_history(k);
            elseif isfield(res, 'trust_radius')
                r = res.trust_radius;
            elseif isfield(res, 'r_tau')
                r = res.r_tau;
            else
                r = 0;
            end
            
            if r > 0
                tr_x = x_best(1) + r * cos(theta);
                tr_y = x_best(2) + r * sin(theta);
                fill(tr_x, tr_y, [0.3 0.9 0.3], 'FaceAlpha', 0.15, ...
                     'EdgeColor', 'none', 'HandleVisibility', 'off');
                plot(tr_x, tr_y, 'Color', [0 0.7 0], 'LineWidth', 2.5, ...
                     'HandleVisibility', 'off');
            end
        end
        
        % -----------------------------------------------------------------
        % Pruning zones (if applicable)
        % -----------------------------------------------------------------
        if methods(m).show_prune
            if isfield(res, 'prune_radius')
                tau = res.prune_radius;
            elseif isfield(res, 'tau')
                tau = res.tau;
            else
                tau = 0;
            end
            
            if tau > 0
                n_show = min(5, size(X_obs, 1));
                for i = 1:n_show
                    pr_x = X_obs(i,1) + tau * cos(theta);
                    pr_y = X_obs(i,2) + tau * sin(theta);
                    plot(pr_x, pr_y, 'r--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
                end
            end
        end
        
        % -----------------------------------------------------------------
        % Observations (colored by value)
        % -----------------------------------------------------------------
        scatter(X_obs(:,1), X_obs(:,2), 60, Y_obs, 'filled', ...
                'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
        
        % -----------------------------------------------------------------
        % True optimum
        % -----------------------------------------------------------------
        if isfield(test, 'x_opt') && ~isempty(test.x_opt)
            for j = 1:size(test.x_opt, 1)
                scatter(test.x_opt(j,1), test.x_opt(j,2), 250, 'h', ...
                        'MarkerFaceColor', [0.2 1 0.2], 'MarkerEdgeColor', 'k', ...
                        'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
        
        % -----------------------------------------------------------------
        % Current best
        % -----------------------------------------------------------------
        scatter(x_best(1), x_best(2), 250, 'p', ...
                'MarkerFaceColor', [1 0.84 0], 'MarkerEdgeColor', 'k', ...
                'LineWidth', 2, 'HandleVisibility', 'off');
        
        % -----------------------------------------------------------------
        % Next sample (if available)
        % -----------------------------------------------------------------
        if n_obs + 1 <= size(res.X_sample, 1)
            x_next = res.X_sample(n_obs + 1, :);
            scatter(x_next(1), x_next(2), 180, 'd', ...
                    'MarkerFaceColor', methods(m).color, 'MarkerEdgeColor', 'k', ...
                    'LineWidth', 2, 'HandleVisibility', 'off');
        end
        
        % -----------------------------------------------------------------
        % Formatting
        % -----------------------------------------------------------------
        xlabel('x_1', 'FontWeight', 'bold');
        ylabel('x_2', 'FontWeight', 'bold');
        title(sprintf('{\\bf%s}', methods(m).name), 'FontSize', 12, ...
              'Color', methods(m).color);
        xlim(xlims); ylim(ylims);
        axis square; grid on;
        set(gca, 'GridAlpha', 0.3);
    end
    
    % =====================================================================
    %  TOP RIGHT: LEGEND PANEL
    % =====================================================================
    subplot(2, n_methods + 1, n_methods + 1);
    axis off;
    
    text(0.05, 0.95, '{\bfLEGEND}', 'FontSize', 13, 'FontWeight', 'bold');
    text(0.05, 0.82, '● Observations', 'FontSize', 10);
    text(0.05, 0.70, '★ Current Best', 'FontSize', 10, 'Color', [0.8 0.6 0]);
    text(0.05, 0.58, '⬢ True Optimum', 'FontSize', 10, 'Color', [0 0.7 0]);
    text(0.05, 0.46, '◆ Next Sample', 'FontSize', 10);
    text(0.05, 0.32, '── Trust Region', 'FontSize', 10, 'Color', [0 0.7 0]);
    text(0.05, 0.20, '- - Pruning Zone', 'FontSize', 10, 'Color', [0.8 0 0]);
    text(0.05, 0.06, '   (first 5 only)', 'FontSize', 8, 'FontAngle', 'italic', ...
     'Color', [0.5 0.5 0.5]);
    
    % =====================================================================
    %  BOTTOM LEFT: CONVERGENCE PLOT
    % =====================================================================
    ax_conv = axes('Position', [0.06, 0.08, 0.50, 0.35]);
    hold on; grid on; box on;
    
    legend_handles = [];
    legend_names = {};
    
    for m = 1:n_methods
        err = abs(test.f_opt - methods(m).result.best_so_far(1:k));
        % Prevent log(0) issues
        err = max(err, 1e-16);
        h = plot(1:k, err, '-', 'Color', methods(m).color, 'LineWidth', 2.5);
        scatter(k, err(end), 120, methods(m).color, 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        
        % Mark iterations where sampling occurred outside trust region
        if isfield(methods(m).result, 'outside_TR_iterations')
            outside_iters = find(methods(m).result.outside_TR_iterations(1:k));
            if ~isempty(outside_iters)
                % Make Red circles MUCH larger and more visible
                scatter(outside_iters, err(outside_iters), 200, ...
                        [1 0.5 0], 'o', 'filled', ...
                        'MarkerEdgeColor', [1 1 1], 'LineWidth', 3);
            end
        end
        
        legend_handles(end+1) = h; %#ok<AGROW>
        legend_names{end+1} = methods(m).name; %#ok<AGROW>
    end
    
    set(gca, 'YScale', 'log', 'FontSize', 10);
    xlabel('Iteration', 'FontWeight', 'bold', 'FontSize', 11);
    ylabel('Error |f^* - f_{best}|', 'FontWeight', 'bold', 'FontSize', 11);
    title('{\bfConvergence Comparison}', 'FontSize', 12);
    xlim([0.5, Kmax + 0.5]);
    legend(legend_handles, legend_names, 'Location', 'northeast', 'FontSize', 9);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    
    % =====================================================================
    %  BOTTOM MIDDLE: RUNTIME & EXPLORATION STATS
    % =====================================================================
    ax_runtime = axes('Position', [0.60, 0.08, 0.15, 0.35]);
    hold on; box on;
    
    times = [methods.time];
    b = bar(1:n_methods, times, 0.65);
    b.FaceColor = 'flat';
    for m = 1:n_methods
        b.CData(m,:) = methods(m).color;
    end
    b.EdgeColor = 'k';
    b.LineWidth = 1;
    
    xticks(1:n_methods);
    xticklabels({'Standard', 'Adaptive\n(Uncert)'});
    xtickangle(25);
    ylabel('Time (s)', 'FontWeight', 'bold');
    title('{\bfRuntime}', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 9);
    
    % =====================================================================
    %  BOTTOM RIGHT: STATISTICS PANEL
    % =====================================================================
    ax_stats = axes('Position', [0.78, 0.08, 0.20, 0.35]);
    axis off;
    
    text(0, 0.95, sprintf('{\\bfIteration %d/%d}', k, Kmax), ...
         'FontSize', 12, 'FontWeight', 'bold');
    text(0, 0.87, repmat('─', 1, 30), 'FontSize', 10);
    
    text(0, 0.78, '{\bfCurrent Error:}', 'FontSize', 10);
    
    y_pos = 0.68;
    for m = 1:n_methods
        err = abs(test.f_opt - methods(m).result.best_so_far(k));
        
        % Abbreviate method name for display
        if contains(methods(m).name, 'Standard')
            short_name = 'Standard';
        else
            short_name = 'Adapt-Uncert';
        end
        
        text(0, y_pos, sprintf('%s:', short_name), 'FontSize', 9, ...
             'Color', methods(m).color);
        text(0.70, y_pos, sprintf('%.2e', err), 'FontSize', 9);
        y_pos = y_pos - 0.10;
    end
    
    % Find current leader
    errors = zeros(n_methods, 1);
    for m = 1:n_methods
        errors(m) = abs(test.f_opt - methods(m).result.best_so_far(k));
    end
    [~, best_m] = min(errors);
    
    text(0, 0.20, repmat('─', 1, 30), 'FontSize', 10);
    text(0, 0.12, '{\bfCurrent Leader:}', 'FontSize', 10);
    
    % Use abbreviation for leader display
    if contains(methods(best_m).name, 'Standard')
        leader_name = 'Standard BO';
    else
        leader_name = 'Adaptive (Uncert)';
    end
    
    text(0, 0.02, leader_name, 'FontSize', 10, ...
         'Color', methods(best_m).color, 'FontWeight', 'bold');
    
    % =====================================================================
    %  MAIN TITLE
    % =====================================================================
    sgtitle(sprintf('{\\bf%s} | Iteration %d/%d | f^* = %.4f', ...
                    test.func_name, k, Kmax, test.f_opt), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    drawnow;
    
    % Save GIF frame
    if save_gif
        frame = getframe(fig);
        im = frame2im(frame);
        [A, map] = rgb2ind(im, 256);
        if k == 1
            imwrite(A, map, gif_name, 'gif', 'LoopCount', Inf, 'DelayTime', pause_time);
        else
            imwrite(A, map, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', pause_time);
        end
    end
    
    pause(pause_time);
end

%% ========================================================================
%  FINAL SUMMARY
%  ========================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                      FINAL RESULTS                            ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Method                 │ Final Error      │ Time    │ Rank  ║\n');
fprintf('╠─────────────────────────┼──────────────────┼─────────┼───────╣\n');

final_errors = zeros(n_methods, 1);
for m = 1:n_methods
    final_errors(m) = abs(test.f_opt - methods(m).result.best_so_far(end));
end
[~, rank_order] = sort(final_errors);
ranks = zeros(n_methods, 1);
for r = 1:n_methods
    ranks(rank_order(r)) = r;
end

for m = 1:n_methods
    fprintf('║  %-22s │ %16.4e │ %5.2fs  │  #%d   ║\n', ...
            methods(m).name, final_errors(m), methods(m).time, ranks(m));
end
fprintf('╚════════════════════════════════════════════════════════════════╝\n');

fprintf('\n★ WINNER: %s (Error: %.4e)\n', methods(rank_order(1)).name, ...
        final_errors(rank_order(1)));

if save_gif
    fprintf('\n✓ Animation saved: %s\n', gif_name);
end

fprintf('\n✓ Visualization complete!\n\n');

end