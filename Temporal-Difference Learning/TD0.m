function V = TD0(env,P,gamma,alpha,numEpisodes)

% Initialize aribtrary function V
V = zeros(numel(env.States),1);

% Loop through each episode
for episode = 1:numEpisodes
  
    % Reset environment
    env.reset;
    env.CurrentState = '[3,2]';
    
    % Initial state
    s = state2idx(env,env.CurrentState);
    
    for T = 1:100
        % Choose an action based on the policy
        a = randsample(1:numel(env.Actions),1,true,P(s,:));
        
        % Simulate the action and get the results
        [s_,r,isdone,~] = step(rlMDPEnv(env),a);
        
        % Update V(s)
        V(s) = V(s) + alpha * (r + gamma * V(s_) - V(s));
        
        % Update the state
        s = s_;
        
        % Check if agent reached a terminal state and move on to the next
        % episode if it did
        if isdone
            break
        end
    end
end
end