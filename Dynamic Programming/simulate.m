function simulate(env,P)
% Plot the environment
% (not sure what rlMDPEnv does, but we need to call it on env to be able to
% plot it)
plot(rlMDPEnv(env));

% Get initial state
s = state2idx(env,env.reset);

isdone = false;
while ~isdone
    % Get best action in current state
    [~,a] = max(P(s,:));
    
    pause(0.5);
    
    % Perform best action
    % (again, not sure what rlMDPEnv does, but this is needed to call step)
    [s,~,isdone,~] = step(rlMDPEnv(env),a);
end
end