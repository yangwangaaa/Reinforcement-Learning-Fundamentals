function [P,V] = valueIteration(env,gamma)
% Value Iteration

% Set the convergence threshold
theta = 1e-6;

% Initialize the value function to be zero for all states
V = zeros(numel(env.States),1);

while true
    delta = 0;
    
    % Loop through each state and get the maximum expected return by any
    % action
    for s = 1:numel(env.States)
        % Store current value of V
        v = V(s);
        
        % Update V
        V(s) = max(oneStepLookahead(env,V,s,gamma));
        
        % Calculate the largest difference between the current and previous value
        % functions for the current state
        delta = max(delta,abs(v - V(s)));
    end
    
    % Check if converged
    if delta < theta
        break
    end
end

% Create an optimal deterministic policy based on the calculated value
% function
P = zeros(numel(env.States),numel(env.Actions));

% Loop through each state and select the action with the highest expected
% return in each state
for s = 1:numel(env.States)
    % Get expected returns for each action in the current state
    actionValues = oneStepLookahead(env,V,s,gamma);
    
    % Find the best action and assign it to the policy for the current
    % state
    [~,bestAction] = max(actionValues);
    P(s,bestAction) = 1;
end
end