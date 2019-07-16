function [P,V] = policyIteration(env,gamma)
% Policy Iteration

% Initialize the value function V and the policy P arbitrarily. In this
% case, V is set to zero for each state and P is a policy with equal
% probability of selecting each action.
V = zeros(numel(env.States),1);
P = ones(numel(env.States),numel(env.Actions))/numel(env.Actions);

policyStable = false;
while ~policyStable
    % Evaluate the current policy
    V = policyEvaluation(env,P,V,gamma);
    
    % Improve the current policy
    [P,policyStable] = policyImprovement(env,P,V,gamma);
end
end