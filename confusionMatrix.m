function [ M,oa,pa,ua,kappa ] = confusionMatrix( label, claMap )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

ind = label~=0;
claMap = claMap.*ind;

label(label==0)=[];
claMap(claMap==0)=[];


claIdx_lab = unique(label);
claIdx_map = unique(claMap);
M = zeros(length(claIdx_lab),length(claIdx_lab));


l = length(label);

for i = 1:l
    M(claIdx_lab(claIdx_lab==label(i)),claIdx_map(claIdx_map==claMap(i))) = M(claIdx_lab(claIdx_lab==label(i)),claIdx_map(claIdx_map==claMap(i))) + 1;
end


% overall accuracy
oa = sum(diag(M))./sum(M(:));

% producer accuracy(accuracy for each class)
pa = diag(M)./sum(M,2);%Average accuracy

% user accuracy(in each classified class, the percentage of correct classified)
ua = diag(M)./sum(M,1)';

% kappa coefficient
po = oa;
pe = sum(sum(M,1).*sum(M,2)')/l^2;
kappa = (po-pe)/(1-pe);




end