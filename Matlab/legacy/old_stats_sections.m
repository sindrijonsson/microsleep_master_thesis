% %% 1.a Apply different duration criteria to MS
% 
% % Get the labels per recordings
% [idx, rec] = findgroups(master.id);
% recLabels = splitapply(@(x) {x}, master{:,"wakeED_vs_restMSE"}, idx);
% orgDurs = cellfun(@(x) get_durations(x, params.secPerLabel), recLabels, ...
%                   "UniformOutput",false);
% orgDurStats = calc_stats(cell2mat(orgDurs'));
% orgDurStats.within = sum(cell2mat(orgDurs') >= 3 & cell2mat(orgDurs') <= 15) / length(cell2mat(orgDurs'))  
% % Set labels < 3 seconds to negative class (wake)
% % Used as training data
% % trainY = cellfun(@(x) remove_invalid_labels(x, ...
% %                                            3, ...
% %                                            inf, ...
% %                                            params.secPerLabel, ...
% %                                            params.negClassValue, ...
% %                                            false), ...
% %                 recLabels, "UniformOutput",false);
% % master.(params.trainTargets) = cell2mat(trainY);
% % master.(params.trainTargets) = cell2mat(recLabels); % Train on everything
% 
% % Set labels > 15 seconds to nan (ignore during evaluation)
% % Used as test data
% % testY = cellfun(@(x) remove_invalid_labels(x, ...
% %                                            0, ...
% %                                            15, ...
% %                                            params.secPerLabel, ...
% %                                            params.replaceInvalidPredictions, ...
% %                                            true), ...
% %                 trainY, "UniformOutput",false);
% % master.(params.testTargets) = cell2mat(testY);
% % testY = cellfun(@(x) remove_invalid_labels(x, ...
% %                                            3, ...
% %                                            15, ...
% %                                            params.secPerLabel, ...
% %                                            nan, ...
% %                                            false), ...
% %                 recLabels, "UniformOutput",false);
% % master.(params.testTargets) = cell2mat(testY);
% 
% 
% 
% 
% %%
% trainDurs = cellfun(@(x) get_durations(x, params.secPerLabel), ...
%                   recLabels(ismember(rec,params.split.train)), ...
%                   "UniformOutput",false);
% testDurs = cellfun(@(x) get_durations(x, params.secPerLabel), ...
%                   recLabels(ismember(rec,params.split.test)), ...
%                   "UniformOutput",false);
% 
% trainDurStats = calc_stats(cell2mat(trainDurs'))
% testDurStats = calc_stats(cell2mat(testDurs'))
% 
% figure(100); clf;
% hold on
% histogram(cell2mat(trainDurs'),50,"Normalization","probability","FaceAlpha",0.5);
% histogram(cell2mat(testDurs'),50,"Normalization","probability","FaceAlpha",0.5);
% xlabel("Duration [sec]")
% ylabel("Density")
% legend(["Training set","Test set"])
% 
% 
% % How many MSE vs Wake?
% trainIdx = ismember(master.id, params.split.train);
% trainFractions = struct;
% trainFractions.MSE = sum(master{trainIdx,params.testTargets} == 1) / sum(trainIdx);
% trainFractions.Wake = sum(master{trainIdx,params.testTargets} == -1) / sum(trainIdx);
% trainFractions.nan = 1-(trainFractions.MSE+trainFractions.Wake);
% trainFractions
% 
% testIdx = ismember(master.id, params.split.test);
% testFractions = struct;
% testFractions.MSE = sum(master{testIdx,params.testTargets} == 1) / sum(testIdx);
% testFractions.Wake = sum(master{testIdx,params.testTargets} == -1) / sum(testIdx);
% testFractions.nan = 1-(testFractions.MSE+testFractions.Wake);
% testFractions
% 
% %% 1.x Write labels to mat for transfer learning
% % cellfun(@(x, name) save(sprintf("../edf_data/%s_status.mat",name),"x"), recLabels, rec)
