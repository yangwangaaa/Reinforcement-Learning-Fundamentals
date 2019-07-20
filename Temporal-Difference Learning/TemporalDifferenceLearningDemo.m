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
%% TD Prediction
% % Initialize arbitrary policy
% P = ones(numel(gridWorld.States),numel(gridWorld.Actions))/numel(gridWorld.Actions);
% V_TD0 = TD0(gridWorld,P,1,1,200);
%% TD Control
% [P_SARSA,Q_SARSA] = SARSA(gridWorld,1,1,0.8,200);
% [P_Qlearning,Q_Qlearning] = Qlearning(gridWorld,1,1,0.8,200);
% save('P_Qlearning','P_Qlearning');
%% Simulate
load P_Qlearning.mat
simulate(gridWorld,P_Qlearning);