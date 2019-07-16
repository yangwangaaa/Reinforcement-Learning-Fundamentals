function V = policyEvaluation(env,P,V,gamma)
% Policy Evaluation

% Set the convergence threshold
theta = 1e-6;

while true
    delta = 0;  
    % Loop through each state and calculate its expected return
    % 1. Get the current state
    for s = 1:numel(env.States)
        v = V(s);
        V(s) = 0;
        % 2. Look through each action
        for a = 1:numel(env.Actions)
            % 3. Look through each next state
            for s_ = 1:numel(env.States)
                % Calculate the expected return using the Bellman equation
                V(s) = V(s) + P(s,a) * env.T(s,s_,a) * (env.R(s,s_,a) + gamma*V(s_));
            end
        end
        
        % Calculate the largest difference between the current and previous value
        % functions for the current state
        delta = max(delta,abs(v - V(s)));
    end

    % Check if converged
    if delta < theta
        break
    end
end
end