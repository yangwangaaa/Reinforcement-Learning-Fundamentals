function V = everyVisitMC(env,P,gamma,numEpisodes)
% First-Visit MC 

% Initialize value function to be all zeros
V = zeros(numel(env.States),1);

% Vector to store the returns at each state
R = zeros(numel(env.States),1);

% Vector to store the number of times each state has been visited
numVisits = zeros(numel(env.States),1);

for episode = 1:numEpisodes
    % Variable to store the data of each episode as 
    % [s0 a0 0
    %  s1 a1 r1
    %  ...
    %  sT-1 aT-1  RT]
    episodeData = [];
    
    % Counter to keep track of time step
    T = 0;
    
    % Initial state
    s = state2idx(env,env.reset);
    [~,a] = max(P(s,:));
    
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
        
        % Get the reward at time step t + 1
        r = episodeData(t,3);
        
        % Calculate the return after time step t
        G = gamma*G + r;
        
        % Increment number of visits to s
        numVisits(s) = numVisits(s) + 1;
        
        % Update value function by adding new return to previous
        % returns at state s and averaging
        R(s) = R(s) + G;
        V(s) = R(s)/numVisits(s);
    end
    
    % To help visualize progress (optional)
    if mod(episode,100) == 0
        fprintf('Episode %d\n',episode);
        reshape(V,4,4)
    end
end
end