setup;
%% Prepare quantities to be Layers
if server
Newskincolour = Param('value',Newskincolour,'learningRate',0); Newskincolour.name='Newskincolour';
Tmatrix       = Param('value',Tmatrix,'learningRate',0);       Tmatrix.name='Tmatrix';
end
%% ------------------------------------------------------------------------
celebaimdb.averageImage = [129.1863,104.7624,93.5940];
muim = single(reshape(celebaimdb.averageImage,[1,1,3,1]));
%% -------------------------- CNN -----------------------------------------
images = Input();
images.name = 'images';

actualshading = Input();
actualshading.name = 'actualshading';

actualmasks = Input();
actualmasks.name = 'actualmasks';

% CNN paramters
nfilters = (single([32 64 128 256 512]));
nclass = single(4); % 4 decoders
%---------------------------- CNN -----------------------------------------
[lightingparameters,b,fmel,fblood,predictedShading,specmask] = CNN(images,nfilters,nclass,LightVectorSize,bSize);
%% ------------------------  Scale Parameters -----------------------------
[weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,predictedShading,specmask] = scalingNet(lightingparameters,b,fmel,fblood,predictedShading,specmask,bSize);
%% ------------------------ Illumination Model ----------------------------
% create the illumination model from CIE standard illuminants: A,D,F
[ e ] = illuminationModel(weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm);
%% ------------------------ Camera Model ----------------------------------
[Sr,Sg,Sb] = cameraModel(mu,PC,b,wavelength);
%% ------------------------ light colour ----------------------------------
[lightcolour] = computelightcolour(e,Sr,Sg,Sb);
%% ------------------------ Specularities ---------------------------------
[ Specularities ] = computeSpecularities(specmask,lightcolour);
%% ---------------- Biophysical to spectral reflectance -------------------
[ R_total ] = BiotoSpectralRef(fmel,fblood,Newskincolour);
%% --------------------------- Image Formation ----------------------------
[ rawAppearance,diffuseAlbedo ]  = ImageFormation(R_total, Sr,Sg,Sb,e,Specularities,predictedShading);
%% --------------------------- White Balance ------------------------------
[ImwhiteBalanced] = WhiteBalance(rawAppearance,lightcolour);

%% ------------------------ from raw To sRGB ------------------------------
[T_RAW2XYZ] = findT(Tmatrix,BGrid);
[ sRGBim ] = fromRawTosRGB(ImwhiteBalanced,T_RAW2XYZ);
%% ------------------------ Naming the layers -----------------------------
e.name='e';
Sr.name='Sr';
Sg.name='Sg';
Sb.name='Sb';
fmel.name='fmel';
fblood.name ='fblood';
predictedShading.name ='predictedShading';
specmask.name ='specmask';
lightcolour.name='lightcolour';
Specularities.name ='Specularities';
R_total.name ='R_total';
rawAppearance.name = 'rawAppearance';
diffuseAlbedo.name ='diffuseAlbedo';
ImwhiteBalanced.name ='ImwhiteBalanced';
sRGBim.name ='sRGBim';
T_RAW2XYZ.name = 'T_RAW2XYZ';
BGrid.name ='BGrid';
b.name ='b';
lightingparameters.name ='lightingparameters';
weightA.name='weightA';
weightD.name='weightD';
CCT.name ='CCT';
Fweights.name='Fweights';
%%
scaleRGB = sRGBim.*single(255);  scaleRGB.name ='scaleRGB';
%scaleRGB = sRGBim;  scaleRGB.name ='scaleRGB';
Y1 = ones(size(muim),'single');
Y1 = Y1.*muim;
rgbim = ((scaleRGB) - Y1);
X1 = ones(size(rgbim),'single');
rgbim = rgbim.*X1; rgbim.name = 'rgbim';
%% ------------------------- LOSSES ---------------------------------------
nFGpix = sum(sum(actualmasks,1),2);

% camera loss
% appearance
delta = (images - rgbim ).*actualmasks;

%shading
scale = sum( sum( (actualshading.*predictedShading).*actualmasks, 1), 2) ./ sum( sum( (predictedShading.^2).*actualmasks, 1), 2);

predictedShading = predictedShading.*scale;
alpha = (actualshading - predictedShading).*actualmasks;

% weights
blossweight = 1e-4;  
appweight = 1e-3; % 
Shadingweight = 0.00001; 
sparseweight = 1e-5;
%----------------------
% camera 
priorB = sum(b(:).^2);  priorB.name ='priorB';
priorloss = (priorB(:)).*blossweight; 
ZY = ones(size(priorloss),'single');
priorloss = priorloss.*ZY;
priorloss.name='priorloss';
%----------------------
% L2: appearance loss :
delta = delta ./ nFGpix;
appearanceloss = sum(delta(:).^single(2)).*appweight; 
Y = ones(size(appearanceloss),'single');
appearanceloss = appearanceloss.*Y;
appearanceloss.name ='appearanceloss';
%----------------------
sparsityloss = sum(Specularities(:)).*sparseweight ;
J = ones(size(sparsityloss),'single');
sparsityloss = sparsityloss.*J;
sparsityloss.name = 'sparsityloss';
%----------------------
% shading
shadingloss = sum(alpha(:).^single(2)) .* Shadingweight;
ff = ones(size(shadingloss),'single');
shadingloss = shadingloss.*ff;
shadingloss.name ='shadingloss';
% ---------------------
loss = appearanceloss + priorloss +shadingloss+sparsityloss;
loss.name= 'loss';
loss.sequentialNames();
%%
if server
  gpuDevice(1);
  images.gpu=true;
  opts.gpus = [1] ;
end
opts.batchSize = batchSize ;
opts.numEpochs = 200 ;
opts.learningRate =1e-3;


opts.expDir = 'data/SelfSupervisonBioMaps' ;
opts.stats = {'loss','appearanceloss','priorloss','shadingloss','sparsityloss'};

net = Net(loss);
[net, stats] = cnn_train_autonn(net, celebaimdb, @getBatch, opts) ;





