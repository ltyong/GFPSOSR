function [fitness_gbest,gbest,record] = GFPSOSR(F,...
    MaxFES,sizepop,D,...
    v_max,v_min,x_max,x_min,...
    c1,c2,a,w_max,w_min)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GFPSOSR
%
% Speed update mode:                Fractional(r=4) + Social learning.
% Gradient update method:           Post extra update.
% Last place elimination mechanism: Normal distribution.
% -------------------------------------------------------------------------
% Input
%
% F;        Functions to be optimized.
% MaxFES:   Maximum number of fitness evaluation.
% sizepop:  Particle swarm size.
% D:        Problem dimension.
%
% v_max:    Velocity range(Maximum range)(Generally x_max/10).
% v_min:    Velocity range(Minimum range)(Generally x_min/10).
% x_max:    Location range(Maximum range).
% x_min:    Location range(Minimum range).
%
% c1:       Learning factor 1.
% c2:       Learning factor 2.
% a:        Fractional order parameter 'alpha'.
% w_max:    Inertia weight range(Maximum range).
% w_min:    Inertia weight range(Minimum range).
% -------------------------------------------------------------------------
% Output
%
% fitness_gbest:The fitness of the optimal point found by the algorithm.
% gbest:        The location of the optimal point found by the algorithm.
% record:       Record the global optimal fitness for each evolution.
% -------------------------------------------------------------------------   

maxgen = floor(MaxFES/sizepop*50/101); % Convert evolution times

% Initialize historical speed
v2 = zeros(D,sizepop);
v3 = zeros(D,sizepop);
v4 = zeros(D,sizepop);

% Generate initial particles and velocities
x = x_min + rand(D,sizepop) * (x_max-x_min); % x_min~x_max uniform random number
v1 = 2*rand(D,sizepop)-1;  % -1~1 uniform random number

% Calculate fitness
fitness = F(x,0);
[fitness_gbest,i] = min(fitness); % Find the global best fitness value
pbest = x;                        % Record individual optimal location
gbest = x(:,i);                   % Record the optimal position of the group
fitness_pbest = fitness;          % Record individual optimal fitness value

% Iterative optimization
t = 1;
record = zeros(1,maxgen); % Record the optimal position of the population in each evolution
while t <= maxgen
    % Inertia factor update
    w = w_max - t / maxgen * (w_max - w_min);
    
    %-----Social learning and fractional-order updating strategy-----
    % MPbest
    MPbest = mean(pbest,2);
    
    % Sort according to individual optimal fitness
    [~,index]=sort(fitness_pbest,'descend'); 
    x = x(:,index);
    v1 = v1(:,index);
    v2 = v2(:,index);
    v3 = v3(:,index);
    v4 = v4(:,index);
    pbest = pbest(:,index);
    fitness_pbest = fitness_pbest(:,index);
    
    % SPbest
    SPbest = zeros(D,sizepop);
    for i = 1:sizepop
        index = randi([i,sizepop]);
        SPbest(:,i) = pbest(:,index);
    end

    % Velocity update
    r1 = rand(D,sizepop);
    r2 = rand(D,sizepop);
    vv = (a*(1-a)*v2)/2 ... 
        + (a*(1-a)*(2-a)*v3)/6 ...
        + (a*(1-a)*(2-a)*(3-a)*v4)/24;
    v = c1 * r1 .* (SPbest - x) ...
        + c2 * r2 .* (MPbest-x) ...
        + (w-1+a) * v1 + vv;
    % Limit velocity
    v = max(v_min,min(v_max,v));
    % Update historical speed
    v4 = v3;
    v3 = v2;
    v2 = v1;
    v1 = v;

    % Location update
    x = x + v;
    % Limit location
    x = max(x_min,min(x_max,x));

    % Calculate fitness value and gradient
    [fitness,g] = F(x,1);

    % Individual optimal update
    index = fitness < fitness_pbest;
    fitness_pbest(index) = fitness(index);
    pbest(:,index) = x(:,index);

    % Group optimal update
    [~,i] = min(fitness);
    if fitness(i) < fitness_gbest
        gbest = x(:,i);
        fitness_gbest = fitness(i);
    end

    %-----Gradient search strategy-----
    % Gradient update
    r1 = rand(D,sizepop);
    v0 = - w * r1 .* abs(v1) .* sign(g);
    x = x + v0;
    v1 = v1 + v0;

    x = max(x_min,min(x_max,x));

    % Calculate fitness value
    fitness = F(x,0);
    
    %-----Terminal replacement mechanism-----
    [worst,index]=max(fitness);
    MU=0.5*(pbest(:,index)+gbest);
    SIGMA=abs(pbest(:,index)-gbest);
    NewX = normrnd(MU,SIGMA);
    NewX = max(x_min,min(x_max,NewX));
    fNewX = F(NewX,0);

    if fNewX<worst
        x(:,index) = NewX;
        fitness(index)=fNewX;
    end

    % Individual optimal update
    index = fitness < fitness_pbest;
    fitness_pbest(index) = fitness(index);
    pbest(:,index) = x(:,index);

    % Group optimal update
    [~,i] = min(fitness);
    if fitness(i) < fitness_gbest
        gbest = x(:,i);
        fitness_gbest = fitness(i);
    end
    
    % Record each global optimal fitness
    record(t) = fitness_gbest;

    t = t + 1;
end

end
