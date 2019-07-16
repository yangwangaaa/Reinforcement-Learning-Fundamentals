function actionValues = oneStepLookahead(env,V,s,gamma)
% One-Step Lookahead
% Initialize action values to be zero
actionValues = zeros(numel(env.Actions),1);

% Loop through each action and calculate the return of taking that
% action in the current state
% 1. Look through each action
for a = 1:numel(env.Actions)
    % 2. Look through each next state
    for s_ = 1:numel(env.States)
        % Calculate the expected value using the Bellman equation
        % and update actionValues
        actionValues(a) = actionValues(a) + env.T(s,s_,a) * (env.R(s,s_,a) + gamma*V(s_));
    end
end
end