function [P,Q] = monteCarloES(env,gamma,numEpisodes)
% Monte Carlo with Exploring Starts

% Initialize arbitrary policy
P = ones(numel(env.States),numel(env.Actions))/numel(env.Actions);

% Variable to store state-action pair values
Q = zeros(numel(env.States),numel(env.Actions));

% Vector to store the returns at each state
R = zeros(numel(env.States),numel(env.Actions));

for episode = 1:numEpisodes
    % Variable to store the data of each episode as
    % [s0 a0 0
    %  s1 a1 r1
    %  ...
    %  sT-1 aT-1  RT]
    episodeData = [];
    
    % Counter to keep track of time step
    T = 0;
    
    % Randomly choose initial state-action pair
    s = state2idx(env,env.reset);
    a = ceil(numel(env.Actions)*rand(1));
    
    % Variable to store the return after time step t
    G = 0;
    
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
        
        % Get best action in current state
        [~,bestActions] = find(P(s,:) == max(P(s,:)));
        a = bestActions(ceil(numel(bestActions)*rand(1)));
    end
    
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
        
        % Get all states visited prior to time step t
        visitedSAPairs = episodeData(1:t - 1,1:2);
        
        % Check if the state-action pair has not been visited
        if ~ismember([s a],visitedSAPairs,'rows')
            % Update state-action value function by adding new return to 
            % previous returns at (s,a) and averaging
            R(s,a) = R(s,a) + G;
            Q(s,a) = R(s,a)/episode;
            
            % Greedily update the policy by choosing the action with the
            % higest return
            [~,bestAction] = max(Q(s,:));
            P(s,:) = 0;
            P(s,bestAction) = 1;
        end
    end
    
    % To help visualize progress (optional)
    if mod(episode,100) == 0
        fprintf('Episode %d\n',episode);
        P
    end
end
end