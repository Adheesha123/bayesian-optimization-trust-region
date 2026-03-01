function tests = test_functions()

tests = struct([]);

%% SCHAFFER N.2  (global optimum at (0,0))
tests(end+1).func_name = 'Schaffer N.2';
tests(end).f = @(x1,x2) -(0.5 + (sin(x1.^2 - x2.^2).^2 - 0.5) ./ (1 + 0.001*(x1.^2 + x2.^2)).^2);
tests(end).domain = [-100, 100];
tests(end).f_opt = 0.0;
tests(end).x_opt = [0, 0];
tests(end).grid_size = 100;

%% ACKLEY (global optimum at (0,0))
tests(end+1).func_name = 'Ackley';
tests(end).f = @(x1,x2) -(-20*exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2))) ...
    - exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))) + exp(1) + 20);
tests(end).domain = [-40, 40];
tests(end).f_opt = 0.0;
tests(end).x_opt = [0, 0];
tests(end).grid_size = 100;

%% RASTRIGIN (global optimum at (0,0))
tests(end+1).func_name = 'Rastrigin';
tests(end).f = @(x1,x2) -(20 + x1.^2 + x2.^2 - 10*(cos(2*pi*x1) + cos(2*pi*x2)));
tests(end).domain = [-5.12, 5.12];
tests(end).f_opt = 0.0;
tests(end).x_opt = [0, 0];
tests(end).grid_size = 100;

%% BRANIN (3 global minima => 3 global maxima for the negated f)
a = 1; b = 5.1/(4*pi^2); c = 5/pi; r = 6; s = 10; t = 1/(8*pi);
tests(end+1).func_name = 'Branin';
tests(end).f = @(x1,x2) -(a*(x2 - b*x1.^2 + c*x1 - r).^2 + s*(1-t)*cos(x1) + s);
% Change domain to 2x2 matrix format
tests(end).domain = [-5,  10;    % x1 ∈ [-5, 10]
                      0,  15];   % x2 ∈ [0, 15]

tests(end).f_opt = -0.397887;
tests(end).x_opt = [ ...
    -pi, 12.275; ...
     pi,  2.275; ...
     9.42478, 2.475 ...
];
tests(end).grid_size = 100;

%% BEALE (global optimum at (3, 0.5))
tests(end+1).func_name = 'Beale';
tests(end).f = @(x1,x2) -((1.5 - x1 + x1.*x2).^2 + ...
                         (2.25 - x1 + x1.*x2.^2).^2 + ...
                         (2.625 - x1 + x1.*x2.^3).^2);
tests(end).domain = [-4.5, 4.5];
tests(end).f_opt = 0.0;
tests(end).x_opt = [3, 0.5];
tests(end).grid_size = 100;

%% GOLDSTEIN-PRICE (global optimum at (0, -1))
tests(end+1).func_name = 'Goldstein-Price';
tests(end).f = @(x1,x2) -((1 + (x1 + x2 + 1).^2 .* ...
        (19 - 14*x1 + 3*x1.^2 - 14*x2 + 6*x1.*x2 + 3*x2.^2)) .* ...
        (30 + (2*x1 - 3*x2).^2 .* ...
        (18 - 32*x1 + 12*x1.^2 + 48*x2 - 36*x1.*x2 + 27*x2.^2)));
tests(end).domain = [-2, 2];
tests(end).f_opt = -3.0;
tests(end).x_opt = [0, -1];
tests(end).grid_size = 100;

%% SPHERE (global optimum at (0,0))
tests(end+1).func_name = 'Sphere';
tests(end).f = @(x1,x2) -(x1.^2 + x2.^2);
tests(end).domain = [-2, 2];
tests(end).f_opt = 0.0;
tests(end).x_opt = [0, 0];
tests(end).grid_size = 100;

end
