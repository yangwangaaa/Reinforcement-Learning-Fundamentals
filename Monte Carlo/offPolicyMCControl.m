function [P,Q] = offPolicyMCControl(env,gamma,numEpisodes)
% Off-Policy MC Control

% Variable to store state-action pair values
Q = zeros(numel(env.States),numel(env.Actions));

% Variable to store state-action pair C values for weighted importance
% sampling
C = zeros(numel(env.States),numel(env.Actions));

% Initialize arbitrary policy 
P = ones(numel(env.States),numel(env.Actions))/numel(env.Actions);

% Generate behavior policy so that is has equal chance of selecting all
% actions
b = ones(numel(env.States),numel(env.Actions))/numel(env.Actions);

for episode = 1:numEpisodes
    % Variable to store the data of each episode as
    % [s0 a0 0
    %  s1 a1 r1
    %  ...
    %  sT-1 aT-1  RT]
    episodeData = [];
    
    % Counter to keep track of time step
    T = 0;
    
    % Reset environment
    env.reset;
    env.CurrentState = '[3,2]';
    
    % Set initial state and action
    s = state2idx(env,env.CurrentState);
    a = 1;
    
    % Generate an episode
    while T < 100
        % Perform action
        [s_,r,isdone,~] = step(rlMDPEnv(env),a);

        % Store current episode data
        episodeData(end + 1,:) = [s a r]; %#ok<AGROW>
        
        if isdone
            break
        end
        
        % Update s and T
        s = s_;
        T = T + 1;
        
        % Sample an action (the best action will have the highest
        % probability)
        a = randsample(1:numel(env.Actions),1,true,b(s,:)); 
    end
    
    % Variable to store the return after time step t
    G = 0;
    
    % Variable to store the weights for weighted importance sampling
    W = 1;
    
    % Loop backwards through each step of the episode
    for t = T:-1:1
        % Get the state at time step t
        s = episodeData(t,1);
        
        % Get the action at time step t
        a = episodeData(t,2);
        
        % Get the reward at time step t + 1
        r = episodeData(t,3);
        
        % Calculate the return after time step t
        G = gamma*G + r;
        
        % Update using weighted importance sampling
        C(s,a) = C(s,a) + W;
        Q(s,a) = Q(s,a) + (W/C(s,a)) * (G - Q(s,a));
        
        % Greedily update the policy
        [~,bestAction] = max(Q(s,:));
        P(s,:) = 0;
        P(s,bestAction) = 1;
        
        W = W * P(s,a)/b(s,a);
        
        % Stop running if W = 0
        if W == 0
            break
        end
    end
    
    % To help visualize progress (optional)
    if mod(episode,100) == 0
        fprintf('Episode %d\n',episode);
        Q
        P
    end
end
end