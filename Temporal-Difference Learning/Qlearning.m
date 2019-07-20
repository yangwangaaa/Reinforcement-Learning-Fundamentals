function [P,Q] = Qlearning(env,gamma,alpha,epsilon,numEpisodes)

% Initialize an arbitrary policy
P = ones(numel(env.States),numel(env.Actions))/numel(env.Actions);

% Initialize an arbitrary Q
Q = zeros(numel(env.States),numel(env.Actions));

% Loop through each episode
for episode = 1:numEpisodes
    % Reset environment
    env.reset;
    env.CurrentState = '[3,2]';
    
    % Initial state
    s = state2idx(env,env.CurrentState);
    
    % For each step of the episode
    for T = 1:100       
        % Use epsilon-greedy method to select action
        if rand(1) > epsilon
            [~,a] = max(Q(s,:));
        else
            a = randsample(1:numel(env.Actions),1,true,P(s,:));
        end
        
        % Simulate the action and get the results
        [s_,r,isdone,~] = step(rlMDPEnv(env),a);
        
        % Update Q(s,a)
        [~,bestAction] = max(Q(s_,:));
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s_,bestAction)) - Q(s,a));
        
        % Update the state and action
        s = s_;
        
        % Check if agent reached a terminal state and move on to the next
        % episode if it did
        if isdone
            break
        end
    end
end

% Create a deterministic greedy policy
for s = 1:size(P,1)
    [~,bestAction] = max(Q(s,:));
    P(s,:) = 0;
    P(s,bestAction) = 1;
end
end