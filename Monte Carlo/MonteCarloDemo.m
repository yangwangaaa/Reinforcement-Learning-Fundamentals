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

%% Monte Carlo Prediction
% % Initialize arbitrary policy
% P = ones(numel(gridWorld.States),numel(gridWorld.Actions))/numel(gridWorld.Actions);
% 
% % Solve using first-visit Monte Carlo
% V_FVMC = firstVisitMC(gridWorld,P,1,10000);
% 
% % Solve using every-visit Monte Carlo
% V_EVMC = everyVisitMC(gridWorld,P,1,10000);

%% Monte Carlo Control
% [P_MCES,Q_MCES] = monteCarloES(gridWorld,1,10000);
[P_OPFVMCC,Q_OPFVMCC] = onPolicyFirstVisitMCControl(gridWorld,1,0.1,500000);
save('P_OPFVMCC','P_OPFVMCC')
%% Simulate
load P_OPFVMCC.mat
simulate(gridWorld,P_OPFVMCC);
