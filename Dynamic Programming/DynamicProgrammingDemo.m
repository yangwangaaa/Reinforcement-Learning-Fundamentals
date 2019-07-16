clear; clc
% Create the grid world environment
gridWorld = createGridWorld(4,4);

% Set the start state
gridWorld.CurrentState = '[3,2]';

% Set the terminal states
gridWorld.TerminalStates = ['[1,1]';'[4,4]'];

% Get the total number of states
numStates = numel(gridWorld.States);

% Get total number of actions
numActions = numel(gridWorld.Actions);

% Define the reward matrix
gridWorld.R = -ones(numStates,numStates,numActions);
gridWorld.R(:,state2idx(gridWorld,gridWorld.TerminalStates),:) = 0;

% Adjust state transition matrix so that all probabilities in terminal
% states are zero
gridWorld.T(state2idx(gridWorld,gridWorld.TerminalStates),:,:) = 0;

% Solve the MDP using policy iteration
[P_PI,V_PI] = policyIteration(gridWorld,1);

% Solve the MDP using value iteration
[P_VI,V_VI] = valueIteration(gridWorld,1);

% Simulate the trained agent
simulate(gridWorld,P_VI);