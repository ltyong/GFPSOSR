%% Initialization
close all;
clear;

%% Set Parameters
% rng(12345);       % Set random seed
F = @Ackley;        % Functions to be optimized.

maxgen = 10000;   	% Evolution times
sizepop = 50;       % Particle swarm size
MaxFES = maxgen*sizepop; % Maximum number of fitness evaluation(NFE)

D = 10;             % Problem dimension
a = 0.5;            % Fractional order parameter 'alpha'
c1 = 2;             % Learning factor
c2 = 2;   

x_max =  50;       % Location range
x_min = -x_max;
v_max =  x_max/10; % Velocity range
v_min = -v_max;

w_max = 0.9;        % Inertia factor range
w_min = 0.4;

%% Test GFPSOSR
[fitness_gbest,gbest,record] = GFPSOSR(F,...
    MaxFES,sizepop,D,v_max,v_min,x_max,x_min,...
    c1,c2,a,w_max,w_min);

fitness_gbest

%% Ackley function
function [f, g] = Ackley(x, isGradient)
    if nargin == 1 || isempty(isGradient)
        isGradient = true;
    end

    n = size(x, 1);
    
    sum1 = sum(x .^ 2, 1);
    sum2 = sum(cos(2 * pi * x), 1);
    sqrtTerm = sqrt(1/n*sum1);
    expSqrtTerm = exp(-0.2*sqrtTerm);
    expCosTerm = exp(1/n*sum2);
    f = -20*expSqrtTerm - expCosTerm + exp(1) + 20;
    
    if isGradient
        g1 = (4/n./sqrtTerm) .* (expSqrtTerm  .* x);
        g1(isnan(g1)) = 0;
        g2 = (2*pi/n) * (expCosTerm .* sin(2 * pi * x));
        g2(isnan(g2)) = 0;
        g = g1 + g2;
    else
        g = [];
    end
end