
server = true;
if server
  % setup  
run('../vl_setupnn');
run('../setup_autonn'); 
load('../imdb/celebaimdb.mat');
cmf = load('../util/rgbCMF.mat');
load ('/local/data/Sarah/BioFaceTemplate/util/illF.mat');
load ('../util/illumA.mat');
load ('../util/illumDmeasured.mat');
load ('../util/Newskincolour.mat');
load ('../util/Tmatrix.mat');

nimages = 50765;
stimages =1;
batchSize= 64;

% load your dataset: training, eval, testing..

end
%% ------------------------------------------------------------------------
% PCA model for camera sensitivities
[mu,PC,EVpca] = CameraSensitivityPCA(cmf);
%% predicted vectors sizes
LightVectorSize = single(15);  % 15 paramters of light model
wavelength = single(33); 
bSize = single(2);  % 2 parameters of camera model
%% normalise illuminations

illF           = reshape(illF,[1 1 33 12]);
illumDmeasured = illumDmeasured';
illumDmeasured = reshape(illumDmeasured, [1 1 33 22]);
% A
illumA         = single(illumA./sum(illumA(:)));
% D
illumDNorm = single(zeros(1,1, 33,22));
for i=1:22
    illumDNorm(1,1,:,i) = illumDmeasured(1,1,:,i)./sum(illumDmeasured(1,1,:,i));
end  
% F
illumFNorm = single(zeros(1,1,33,12));
for i=1:12
    illumFNorm(1,1,:,i) = illF(1,1,:,i)./sum(illF(1,1,:,i));
end 
%%
mu = Param('value',mu,'learningRate',0);
PC = Param('value',PC,'learningRate',0);
mu = gpuArray(mu);
PC = gpuArray(PC);
illumA = Param('value',illumA,'learningRate',0);
illumA = gpuArray(illumA);
illumFNorm = Param('value',illumFNorm,'learningRate',0);
illumFNorm = gpuArray(illumFNorm);
