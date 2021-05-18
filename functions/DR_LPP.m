function [M,W]=DR_LPP(TrainSample,k,d,sigma,W)
fea=TrainSample';
      options = [];
      options.NeighborMode = 'KNN';
      options.k = k;
      options.WeightMode = 'HeatKernel';
      options.t = sigma;
%       options.idx = idx;
%       options.Regu=1;
%       options.ReguAlpha = 0.05;
%       W=CreateW(fea,options);
    if ~exist('W','var')
         W = constructW(fea,options);
%          W=normlizedMax(W);
%          W=(W+W')/2;
    end
      options.PCARatio =1;
      options.ReducedDim=d;
      M= LPP(W, options, fea);
end