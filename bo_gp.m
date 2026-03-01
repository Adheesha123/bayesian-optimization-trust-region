function out = bo_gp(test, acq_name, N, n_initial, sigma_n, ell_factor, epsilon, rng_seed)
    % --- 1. Setup ---
    f = test.f;
    dom = test.domain;
    f_opt = test.f_opt;
    func_name = test.func_name;
    grid_size = test.grid_size;
    
    rng(rng_seed);
    
    % --- 2. Robust Domain Parsing ---
    if size(dom, 1) == 2
        xlims = dom(1, :);
        ylims = dom(2, :);
    else
        xlims = dom;
        ylims = dom;
    end
    
    % Create Lower Bound (lb) and Upper Bound (ub) vectors
    lb = [xlims(1), ylims(1)];
    ub = [xlims(2), ylims(2)];
    
    % --- 3. Initial Sampling ---
    X_sample = lb + (ub - lb) .* rand(n_initial, 2);
    Y_sample = f(X_sample(:,1), X_sample(:,2));
    y_max = max(Y_sample);
    
    fprintf('\nInitial samples for %s:\n', func_name);
    fprintf('---------------------------------\n');
    for i = 1:min(n_initial,3)
        fprintf('Init %2d: x=[%8.3f,%8.3f]  y=%10.6f\n', ...
            i, X_sample(i,1), X_sample(i,2), Y_sample(i));
    end
    fprintf('---------------------------------\n');
    
    % --- 4. Grid Setup ---
    [x1g, x2g] = meshgrid(linspace(xlims(1), xlims(2), grid_size), ...
                          linspace(ylims(1), ylims(2), grid_size));
    X_grid = [x1g(:), x2g(:)];
    n_grid = size(X_grid,1);
    
    % --- 5. GP Hyperparameters ---
    ranges = ub - lb;
    l = ell_factor * mean(ranges);
    
    kss = ones(n_grid, 1); 
    acq_fun = get_acq_fun(acq_name);
    
    % --- 6. History Storage ---
    acq_history = zeros(N,1);
    best_so_far = zeros(N,1);
    err_history = zeros(N,1);
    
    % --- 7. Optimization Loop ---
    for iter = 1:N
        % Kernel Matrices
        K  = rbf_kernel(X_sample, X_sample, l) + sigma_n^2 * eye(size(X_sample,1));
        Ks = rbf_kernel(X_sample, X_grid, l);
        
        % Cholesky
        [L, ~] = chol_jitter(K);
        alpha = L' \ (L \ Y_sample);
        
        % Posterior
        mu = Ks' * alpha;
        v = L \ Ks;
        sigma2 = kss - sum(v.^2, 1)'; 
        sigma = sqrt(max(sigma2, 1e-12));
        
        % Acquisition
        acq = acq_fun(mu, sigma, y_max);
       
        % Standard BO theoretically allows re-sampling if variance is non-zero.
        
        [max_acq, idx] = max(acq);
        x_next = X_grid(idx,:);
        
        % Evaluate
        y_next = f(x_next(1), x_next(2));
        if y_next > y_max
            y_max = y_next;
        end
        
        % Update
        X_sample = [X_sample; x_next];
        Y_sample = [Y_sample; y_next];
        
        % Track
        acq_history(iter) = max_acq;
        best_so_far(iter) = y_max;
        err_history(iter) = abs(f_opt - best_so_far(iter));
        
        fprintf('Iter %3d: x=[%8.3f,%8.3f] y=%10.6f best=%10.6f err=%10.6e\n', ...
            iter, x_next(1), x_next(2), y_next, best_so_far(iter), err_history(iter));
        
        % ==================================================================
        % STOPPING CRITERIA 
        % ==================================================================
        % if err_history(iter) < epsilon
        %     fprintf('   -> Convergence threshold met (err < %g) at iter %d\n', epsilon, iter);
        %     % To strictly stop early, uncomment the lines below:
        %     % best_so_far = best_so_far(1:iter);
        %     % err_history = err_history(1:iter);
        %     % acq_history = acq_history(1:iter);
        %     % break; 
        % end
    end
    
    [y_best, idx_best] = max(Y_sample);
    x_best = X_sample(idx_best,:);
    
    out.func_name   = func_name;
    out.x_best      = x_best;
    out.y_best      = y_best;
    out.best_so_far = best_so_far;
    out.err_history = err_history;
    out.acq_history = acq_history;
    out.X_sample    = X_sample;
    out.Y_sample    = Y_sample;
    out.domain      = dom;
    out.f_opt       = f_opt;
end
%% Helpers
function K = rbf_kernel(X1, X2, l)
    D2 = sqdist(X1, X2);
    K = exp(-0.5 * D2 / (l^2));     
end
function D2 = sqdist(X1, X2)
    n1 = size(X1,1); n2 = size(X2,1);
    D2 = sum(X1.^2,2)*ones(1,n2) + ones(n1,1)*sum(X2.^2,2)' - 2*(X1*X2');
    D2 = max(D2,0);
end
function [L, jitter_used] = chol_jitter(K)
    jitter = 1e-10; max_tries = 10;
    for t = 1:max_tries
        [L, p] = chol(K + jitter*eye(size(K)), 'lower');
        if p == 0, jitter_used = jitter; return; end
        jitter = jitter * 10;
    end
    error('Cholesky failed.');
end
function acq_fun = get_acq_fun(acq_name)
    acq_name = upper(string(acq_name));
    switch acq_name
        case "EI",  acq_fun = @(mu, sig, ymax) ei(mu, sig, ymax);
        otherwise,  error("Only EI implemented for brevity here.");
    end
end
function a = ei(mu, sigma, y_max)
    sigma_safe = max(sigma, 1e-12);
    z = (mu - y_max) ./ sigma_safe;
    a = (mu - y_max).*normcdf(z) + sigma.*normpdf(z);
    a(sigma < 1e-12) = 0;
end