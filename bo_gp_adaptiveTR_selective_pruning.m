function out = bo_gp_adaptiveTR_selective_pruning(test, acq_name, N, n_initial, sigma_n, ...
    ell_factor, alpha_prune, beta_trust, epsilon, rng_seed)
% BO_GP_M_ADAPTIVE - Adaptive Trust Region Bayesian Optimization
%
% =========================================================================
%                           ALGORITHM DESCRIPTION
% =========================================================================
%
% KEY DIFFERENCE FROM FIXED MODIFIED BO:
%   - Trust Region MOVES and EXPANDS based on improvement
%   - After 3 consecutive failures, samples OUTSIDE TR in high uncertainty regions
%   - Trust Region is NEVER REMOVED
%
% STEP-BY-STEP PROCESS:
%   1. Sample n_initial random points, evaluate them
%   2. Find best point so far (x_best, y_max)
%   3. Place initial Trust Region around x_best with radius r_tau
%   4. For each iteration:
%
%      a. Track consecutive failures (no improvement)
%      
%      IF failure_count < 3:
%        - Active Set = INSIDE trust region AND OUTSIDE pruning zones
%        - Compute GP posterior on active set
%        - Select x_next = argmax(EI)
%        - Evaluate y_next = f(x_next)
%        
%        IF y_next > y_max (IMPROVEMENT):
%          - Update x_best = x_next, y_max = y_next
%          - MOVE trust region center to new x_best
%          - EXPAND trust region: r_tau = min(r_tau * expand_factor, r_tau_max)
%          - Reset failure_count = 0
%        ELSE (NO IMPROVEMENT):
%          - Increment failure_count
%
%      IF failure_count >= 3:
%        - Active Set = OUTSIDE trust region AND OUTSIDE pruning zones
%        - Compute GP posterior on active set
%        - Select x_next = argmax(uncertainty) [use sigma directly]
%        - Evaluate y_next
%        - Update global best if improved
%        - Reset failure_count = 0
%        - Move TR to current global best
%
%   5. Continue until max iterations or convergence
%
% PARAMETERS:
%   r_tau_init = beta_trust * l     (Initial trust region radius)
%   tau = alpha_prune * l           (Pruning radius, fixed)
%   expand_factor = 1.2             (TR expansion on improvement)
%   r_tau_max = 2 * r_tau_init      (Maximum TR radius - HARD CAP)
%   failure_threshold = 3           (Failures before exploring outside TR)
%
% =========================================================================
%
% INPUTS:
%   test        - Test function struct
%   acq_name    - Acquisition function ('EI')
%   N           - Maximum iterations
%   n_initial   - Initial samples
%   sigma_n     - Noise variance
%   ell_factor  - Length scale factor
%   alpha_prune - Pruning radius factor
%   beta_trust  - Initial trust region factor
%   epsilon     - Error threshold
%   rng_seed    - Random seed
%
% OUTPUTS:
%   out.x_best          - Best point found
%   out.y_best          - Best value found
%   out.best_so_far     - Best value history
%   out.err_history     - Error history
%   out.r_tau_history   - Trust region radius history
%   out.failure_history - Consecutive failure count history
%   out.X_sample        - All sampled points
%   out.Y_sample        - All function values
%
% =========================================================================

    %% ====================================================================
    %  1. SETUP
    %  ====================================================================
    f = test.f;
    dom = test.domain;
    f_opt = test.f_opt;
    func_name = test.func_name;
    grid_size = test.grid_size;
    
    rng(rng_seed);
    
    % Parse domain
    if size(dom, 1) == 2
        xlims = dom(1, :);
        ylims = dom(2, :);
    else
        xlims = dom;
        ylims = dom;
    end
    lb = [xlims(1), ylims(1)];
    ub = [xlims(2), ylims(2)];
    
    % Length scale
    ranges = ub - lb;
    l = ell_factor * mean(ranges);
    
    % Create grid
    [x1g, x2g] = meshgrid(linspace(xlims(1), xlims(2), grid_size), ...
                          linspace(ylims(1), ylims(2), grid_size));
    X_grid = [x1g(:), x2g(:)];
    
    %% ====================================================================
    %  2. ADAPTIVE PARAMETERS
    %  ====================================================================
    tau = alpha_prune * l;              % Pruning radius (FIXED)
    r_tau_init = beta_trust * l;        % Initial trust region radius
    r_tau_max = 2.0 * r_tau_init;       % Maximum trust region radius (HARD CAP)
    expand_factor = 1.2;                % Expansion factor on improvement
    failure_threshold = 3;              % Number of failures before exploring outside TR
    
    r_tau = r_tau_init;                 % Current trust region radius
    failure_count = 0;                  % Consecutive failures counter
    
    fprintf('\n=== Adaptive BO (Trust Region with Failure-Based Exploration) ===\n');
    fprintf('Initial TR Radius:   %.4f\n', r_tau_init);
    fprintf('Max TR Radius (CAP): %.4f\n', r_tau_max);
    fprintf('Pruning Radius:      %.4f\n', tau);
    fprintf('Expand Factor:       %.2f\n', expand_factor);
    fprintf('Failure Threshold:   %d\n', failure_threshold);
    fprintf('==================================================================\n\n');
    
    %% ====================================================================
    %  3. INITIAL SAMPLING
    %  ====================================================================
    X_sample = lb + (ub - lb) .* rand(n_initial, 2);
    Y_sample = f(X_sample(:,1), X_sample(:,2));
    
    % Find initial best
    [y_max, idx_best] = max(Y_sample);
    x_best = X_sample(idx_best, :);
    
    fprintf('Initial best: f(%.3f, %.3f) = %.6f\n\n', x_best(1), x_best(2), y_max);
    
    %% ====================================================================
    %  4. STORAGE
    %  ====================================================================
    best_so_far = zeros(N, 1);
    err_history = zeros(N, 1);
    r_tau_history = zeros(N, 1);        % Track TR radius over time
    failure_history = zeros(N, 1);      % Track consecutive failures
    mode_history = cell(N, 1);          % Track mode for debugging

    % Track which points are improving vs non-improving
    improving_mask = false(N + n_initial, 1);  % Preallocate logical array
    improving_mask(idx_best) = true;            % Initial best is improving

    acq_fun = @(mu, sig, ymax) ei(mu, sig, ymax);
    
    
    %% ====================================================================
    %  5. MAIN OPTIMIZATION LOOP
    %  ====================================================================
    for iter = 1:N
        
        % ==================================================================
        % STEP A: Compute distances
        % ==================================================================
        d_to_best = sqrt(sum((X_grid - x_best).^2, 2));
        D_to_obs = euclid_dist(X_grid, X_sample);
        d_to_closest = min(D_to_obs, [], 2);
        
        % ==================================================================
        % STEP B: Define Active Set based on failure count
        % ==================================================================
        % SELECTIVE PRUNING: Only prune around NON-improving points
        n_current = size(X_sample, 1);
        non_improving_mask = ~improving_mask(1:n_current);

        if any(non_improving_mask)
            X_non_improving = X_sample(non_improving_mask, :);
            D_to_non_improving = euclid_dist(X_grid, X_non_improving);
            d_to_closest_non_improving = min(D_to_non_improving, [], 2);
            mask_outside_prune = (d_to_closest_non_improving >= tau);
        else
            % No non-improving points yet, so no pruning
            mask_outside_prune = true(size(X_grid, 1), 1);
        end
        mask_not_sampled = (d_to_closest > 1e-10);
        
        if failure_count < failure_threshold
            % ============================================================
            % NORMAL MODE: Explore INSIDE trust region
            % Active = INSIDE trust region AND OUTSIDE pruning zones
            % ============================================================
            mask_in_trust = (d_to_best <= r_tau);
            mask_active = mask_in_trust & mask_outside_prune & mask_not_sampled;
            
            % Fallback if too few points
            if sum(mask_active) < 3
                mask_active = mask_in_trust & mask_not_sampled;
            end
            if sum(mask_active) < 3
                % Trust region too small, expand search slightly
                mask_active = mask_outside_prune & mask_not_sampled;
            end
            
            mode_str = sprintf('TR (r=%.2f, fails=%d)', r_tau, failure_count);
            use_uncertainty = false;
            
        else
            % ============================================================
            % EXPLORATION MODE: Sample OUTSIDE trust region
            % Active = OUTSIDE trust region AND OUTSIDE pruning zones
            % Use pure uncertainty (sigma) instead of EI
            % ============================================================
            mask_outside_trust = (d_to_best > r_tau);
            mask_active = mask_outside_trust & mask_outside_prune & mask_not_sampled;
            
            if sum(mask_active) < 3
                % If not enough points outside TR, just avoid pruning zones
                mask_active = mask_outside_prune & mask_not_sampled;
            end
            
            mode_str = sprintf('EXPLORE (r=%.2f, fails=%d)', r_tau, failure_count);
            use_uncertainty = true;
        end
        
        X_active = X_grid(mask_active, :);
        
        % ==================================================================
        % STEP C: GP Posterior
        % ==================================================================
        K = rbf_kernel(X_sample, X_sample, l) + sigma_n * eye(size(X_sample,1));
        [L_chol, ~] = chol_jitter(K);
        Ks = rbf_kernel(X_sample, X_active, l);
        
        alpha_gp = L_chol' \ (L_chol \ Y_sample);
        mu = Ks' * alpha_gp;
        
        v = L_chol \ Ks;
        kss = ones(size(X_active, 1), 1);
        sigma2 = kss - sum(v.^2, 1)';
        sigma = sqrt(max(sigma2, 1e-12));
        
        % ==================================================================
        % STEP D: Acquisition Function
        % ==================================================================
        if use_uncertainty
            % Pure uncertainty-based exploration
            acq_vals = sigma;
        else
            % Standard EI
            acq_vals = acq_fun(mu, sigma, y_max);
        end
        
        [~, idx_max] = max(acq_vals);
        x_next = X_active(idx_max, :);
        y_next = f(x_next(1), x_next(2));
        
        % ==================================================================
        % STEP E: Update samples
        % ==================================================================
        X_sample = [X_sample; x_next];
        Y_sample = [Y_sample; y_next];
        
        % ==================================================================
        % STEP F: Adaptive Trust Region Logic
        % ==================================================================
        if failure_count < failure_threshold
            % ----------------------------------------------------------
            % NORMAL MODE: Check if we improved
            % ----------------------------------------------------------
            if y_next > y_max
                % ======================================================
                % IMPROVEMENT! 
                % - Update best
                % - Move TR center to new best
                % - Expand TR (with cap)
                % - Reset failure counter
                % ======================================================
                y_max = y_next;
                x_best = x_next;
                r_tau_old = r_tau;
                r_tau = min(r_tau * expand_factor, r_tau_max);
                failure_count = 0;
                improving_mask(n_initial + iter) = true;  % Mark as improving (no pruning)
                
                fprintf('Iter %3d [%s]: IMPROVED! y=%.4f | TR %.2f->%.2f\n', ...
                    iter, mode_str, y_next, r_tau_old, r_tau);
            else
                % ======================================================
                % NO IMPROVEMENT
                % - Increment failure counter
                % ======================================================
                failure_count = failure_count + 1;
                
                fprintf('Iter %3d [%s]: No improvement (y=%.4f)\n', ...
                    iter, mode_str, y_next);
            end
        else
            % ----------------------------------------------------------
            % EXPLORATION MODE: We sampled outside TR
            % ----------------------------------------------------------
            % Update global best if this exploration found something better
            [y_max_new, idx_best_new] = max(Y_sample);
            
            if y_max_new > y_max
                % Found better point during exploration
                y_max = y_max_new;
                x_best = X_sample(idx_best_new, :);
                improving_mask(n_initial + iter) = true;  % Mark exploration point as improving
                fprintf('Iter %3d [%s]: Found better point! y=%.4f | Moving TR to (%.2f,%.2f)\n', ...
                    iter, mode_str, y_max, x_best(1), x_best(2));
            else
                fprintf('Iter %3d [%s]: Explored (y=%.4f) | Resetting to TR mode\n', ...
                    iter, mode_str, y_next);
            end
            
            % Reset failure counter and return to normal mode
            failure_count = 0;
        end
        
        % ==================================================================
        % STEP G: Record history
        % ==================================================================
        best_so_far(iter) = y_max;
        err_history(iter) = abs(f_opt - y_max);
        r_tau_history(iter) = r_tau;
        failure_history(iter) = failure_count;
        mode_history{iter} = mode_str;
        
        % ==================================================================
        % STEP H: Check stopping criteria
        % ==================================================================
        % if err_history(iter) < epsilon
        %     fprintf('\n*** Convergence achieved at iteration %d ***\n', iter);
        %     break;
        % end
    end
    
    %% ====================================================================
    %  6. OUTPUT
    %  ====================================================================
    out.func_name       = func_name;
    out.x_best          = x_best;
    out.y_best          = y_max;
    out.best_so_far     = best_so_far;
    out.err_history     = err_history;
    out.X_sample        = X_sample;
    out.Y_sample        = Y_sample;
    out.r_tau           = r_tau_init;       % Initial value
    out.r_tau_history   = r_tau_history;    % Full history
    out.failure_history = failure_history;  % Failure count history
    out.tau             = tau;
    out.domain          = dom;
    out.mode_history    = mode_history;
    out.improving_mask  = improving_mask(1:n_initial+N);  % Which points improved
    out.n_improving     = sum(improving_mask(1:n_initial+N));
    out.n_non_improving = (n_initial + N) - sum(improving_mask(1:n_initial+N));
end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function K = rbf_kernel(X1, X2, l)
    D2 = sqdist(X1, X2);
    K = exp(-0.5 * D2 / (l^2));
end

function D2 = sqdist(X1, X2)
    n1 = size(X1,1); 
    n2 = size(X2,1);
    D2 = sum(X1.^2,2)*ones(1,n2) + ones(n1,1)*sum(X2.^2,2)' - 2*(X1*X2');
    D2 = max(D2, 0);
end

function D = euclid_dist(X1, X2)
    D = sqrt(sqdist(X1, X2));
end

function [L, jitter_used] = chol_jitter(K)
    jitter = 1e-10;
    max_tries = 10;
    for t = 1:max_tries
        [L, p] = chol(K + jitter*eye(size(K)), 'lower');
        if p == 0
            jitter_used = jitter;
            return;
        end
        jitter = jitter * 10;
    end
    error('Cholesky decomposition failed.');
end

function a = ei(mu, sigma, y_max)
    sigma_safe = max(sigma, 1e-12);
    z = (mu - y_max) ./ sigma_safe;
    a = (mu - y_max) .* normcdf(z) + sigma_safe .* normpdf(z);
    a(sigma < 1e-12) = 0;
end