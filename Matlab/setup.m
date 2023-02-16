%% Setup
addpath(genpath("utils\"));
addpath(genpath("data\"));
addpath(genpath("analysis\"))
addpath(genpath("training_performance\"));
addpath(genpath("dev\"));

% Initalize nice matlab colors
cc = lines(8);
colors = struct;
colors.BLUE = cc(1,:);
colors.ORANGE = cc(2,:);
colors.YELLOW = cc(3,:);
colors.PURPLE = cc(4,:);
colors.GREEN = cc(5,:);
colors.CYAN = cc(6,:);
colors.RED = cc(7,:);

AASM = ["WAKE","N1","N2","N3","REM"];

mdls = ["mU-Sleep"
        "patU-Sleep";
        "mU-SSL";
        "FB-LSTM";
        "FB-RF";
        "FB-SVM";
        "CNN-16s"];

time = linspace(0, 40, 40*60);