function [P,policyStable] = policyImprovement(env,P,V,gamma)
% Policy Improvement

% Variable to check whether or not optimal policy has been reached
% - If policy changes, set to false; otherwise, set to true
policyStable = true;

% Loop through each state and greedily update the policy by choosing
% the best action
for s = 1:numel(env.States)
    % Find the best action under the current policy in the current
    % state
    [~,bestCurrentAction] = max(P(s,:));
    
    % Find the best action by looking one step ahead
    actionValues = oneStepLookahead(env,V,s,gamma);
    [~,bestAction] = max(actionValues);
    
    % Check if optimal policy has been reached by checking if the
    % policy has changed
    if bestCurrentAction ~= bestAction
        policyStable = false;
    end
    % Greedily update the current policy by selecting the action that
    % yields the highest return
    P(s,:) = 0;
    P(s,bestAction) = 1;
end
end