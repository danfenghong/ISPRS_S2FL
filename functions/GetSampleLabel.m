function [TrainSample, TestSample, TrainLabel, TestLabel,TestLocation]=GetSampleLabel(ImageData,TrainImage,TestImage)

SpectralData=hyperConvert2d(ImageData);
TrainIndex=hyperConvert2d(TrainImage);
TestIndex=hyperConvert2d(TestImage);
TrainSample=[];
TestSample=[];
TrainLabel=[];
TestLabel=[];
TestLocation=[];

for i=1:max(TrainIndex)
    gTR=find(TrainIndex==i);
    TrainSample=[TrainSample,SpectralData(:,gTR)];
    TrainLabel=[TrainLabel,TrainIndex(:,gTR)];
    gTE=find(TestIndex==i);
    TestSample=[TestSample,SpectralData(:,gTE)]; 
    TestLabel=[TestLabel,TestIndex(:,gTE)];
    TestLocation=[TestLocation,gTE];
end

end