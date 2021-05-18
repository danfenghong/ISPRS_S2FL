clc;
clear;
close all;

%input data
load data_HS_LR.mat;
load data_MS_HR.mat;
TrainImage = double(imread('2013_IEEE_GRSS_DF_Contest_Samples_TR.tif'));
TestImage = double(imread('2013_IEEE_GRSS_DF_Contest_Samples_VA.tif'));

[m, n, z] = size(data_HS_LR);

HSI2d = hyperConvert2d(data_HS_LR);
MSI2d = hyperConvert2d(data_MS_HR);

TR2d = hyperConvert2d(TrainImage);
TE2d = hyperConvert2d(TestImage);

TrainLabel = TR2d(:, TR2d > 0);
TestLabel = TE2d(:, TE2d > 0);

% normalization
for i = 1 : size(HSI2d, 1)
    HSI2d(i, :) = mat2gray(HSI2d(i, :));
end
for i = 1 : size(MSI2d, 1)
    MSI2d(i, :) = mat2gray(MSI2d(i, :));
end

traindata_SP_hsi = HSI2d(:, TR2d > 0);
testdata_SP_hsi = HSI2d(:, TE2d > 0);

traindata_SP_msi = MSI2d(:, TR2d > 0);
testdata_SP_msi = MSI2d(:, TE2d > 0);

SP_1 = [traindata_SP_hsi; zeros(size(traindata_SP_msi))];
SP_2 = [zeros(size(traindata_SP_hsi)); traindata_SP_msi];

SP_3 = [testdata_SP_hsi; zeros(size(testdata_SP_msi))];
SP_4 = [zeros(size(testdata_SP_hsi)); testdata_SP_msi];

W_SP1 = creatLap(traindata_SP_hsi, 45, 1); % CoSpace-l2: 10, CoSpace-l1: 10
W_SP2 = creatLap(traindata_SP_msi, 45, 1); % CoSpace-l2: 10, CoSpace-l1: 10 

    %generate graph matrix(W) and Laplacian matrix(L)
    dis = pdist([TrainLabel,TrainLabel]');
    dis = squareform(dis);
    dis(dis>0) = -1;
    dis(dis==0) = 1;
    dis(dis<0) = 0;
    dis(1 : 2832, 1 : 2832) = W_SP1;
    dis(1 + 2832 : end, 1 + 2832 : end) = W_SP2;
    W = 2 * size(dis,2) * dis / sum(sum(dis));
    L = diag(sum(W)) - W;
    
    %SMA initialization
    LP = DR_LPP([SP_1, SP_2], 10, 30, 1, W);
    Y = OneHotEncoding(TrainLabel, max(TrainLabel));
    
    % S2FL model
    [theta00, theta12, P, res] = S2FL([Y, Y], [SP_1, SP_2], LP' * [SP_1, SP_2],0.01, 0.1, L, 1000);
    f_hsi = (theta00 + theta12) * [SP_1, SP_2];
    f_msi = (theta00 + theta12) * [SP_3, SP_4];
    
    % CoSpace-l2
%     [theta,P,res] = CoSpace([Y, Y], [SP_1, SP_2], LP' * [SP_1, SP_2], 1, 0.01, L, 1000);
%     f_hsi = theta * [SP_1, SP_2];
%     f_msi = theta * [SP_3, SP_4];
    
    % CoSpace-l1
%     [theta,P,res] = CoSpace_l1([Y, Y], [SP_1, SP_2], LP' * [SP_1, SP_2], 0.1, 0.1, L, 1000);     
%     f_hsi = theta * [SP_1, SP_2];
%     f_msi = theta * [SP_3, SP_4];
    
    traindata = [f_hsi(:, 1 : size(SP_1, 2)) + f_hsi(:, 1 + size(SP_1, 2) : end)];
    testdata = [f_msi(:, 1 : size(SP_3, 2)) + f_msi(:, 1 + size(SP_3, 2) : end)];    
    
%% NN classifier with Euclidean distance 
mdl = ClassificationKNN.fit(traindata', TrainLabel', 'NumNeighbors', 1, 'distance', 'euclidean'); 
characterClass = predict(mdl, testdata'); 
[~, oa, pa, ua, kappa] = confusionMatrix(TestLabel, characterClass');
