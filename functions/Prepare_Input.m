function [X_1, X_2, Y, W, L, Z] = Prepare_Input(TrainMS, TrainHS, TrainLabel, param, isClustering, num)

%% Input:
%         TrainMS         - training samples for multispectral data,each 
%                           column vectro of the data is a sample vector
%         TrainHS         - training samples for hyperspectral data,each 
%                           column vectro of the data is a sample vector
%         TrainLabel      - labels for training samples
%         param           - parameters for LPP
%                         - k     : the number of neighbor
%                         - d     : subspace dimension
%                         - sigma : standard deviation for Gaussian kernel
%         isClustering    - 'True' : run the clustering for large-scale data
%                         - 'False': output the original data
%         num             - specify the number of cluster center if isClustering is True    

%% Ouput:
%         X_1             - one modality for model input
%         X_2             - another modality for model input
%         Y               - one-hot encoded label matrix
%         W               - graph matrix
%         L               - Laplacian matrix
%         Z               - initialized latent subspace obtained by LPP

if strcmp(isClustering, 'False')
    
    X_1 = [TrainHS;zeros(size(TrainMS))];
    X_2 = [zeros(size(TrainHS));TrainMS];
    
    %one hot encoding for training labels
    Y = OneHotEncoding(TrainLabel,max(TrainLabel));
    
    %generate graph matrix(W) and Laplacian matrix(L)
    dis = pdist([TrainLabel,TrainLabel]');
    dis = squareform(dis);
    dis(dis>0) = -1;
    dis(dis==0) = 1;
    dis(dis<0) = 0;
    W = 2 * size(dis,2) * dis / sum(sum(dis));
    L = diag(sum(W)) - W;
    
    LP = DR_LPP([X_1,X_2],param.k,param.d,param.sigma,W); %LP: linear projections learned by LPP
    Z = LP' * [X_1,X_2];    
end

if strcmp(isClustering, 'True')
    
    if nargin < 6
        error('Please specify the number of cluster center (num)!')
    end
    
    %feature stacking
    Train_HSMS=[TrainHS;TrainMS];
    
    %generate cluster centers and corresponding labels
    [Train_HSMS_CC,TrainLabel_CC]=Create_Cluster_Center(Train_HSMS,TrainLabel,num);
    
    %cluster centers for HS and MS, respectively
    TrainHS_CC = Train_HSMS_CC(1:size(TrainHS,1),:);
    TrainMS_CC = Train_HSMS_CC(size(TrainHS,1)+1:end,:);
    
    X_1 = [TrainHS_CC;zeros(size(TrainMS_CC))];
    X_2 = [zeros(size(TrainHS_CC));TrainMS_CC];

    %one hot encoding for training labels
    Y = OneHotEncoding(TrainLabel_CC,max(TrainLabel_CC));
    
    %generate graph matrix(W) and Laplacian matrix(L)
    dis = pdist([TrainLabel_CC,TrainLabel_CC]');
    dis = squareform(dis);
    dis(dis>0) = -1;
    dis(dis==0) = 1;
    dis(dis<0) = 0;
    W = 2 * size(dis,2) * dis / sum(sum(dis));
    L = diag(sum(W)) - W;
    
    LP = DR_LPP([X_1,X_2],param.k,param.d,param.sigma,W); %LP: linear projections learned by LPP
    Z = LP' * [X_1,X_2]; 
end