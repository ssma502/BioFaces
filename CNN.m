function [lightingparameters,b,fmel,fblood,Shading,specmask] = CNN(images,nfilters,nclass,LightVectorSize,bSize)
%Inputs:,                                           
%     images           : input to CNN
%     nfilters         : vector of filter sizes
%     nclass           : number of output maps
%     lightVectorLSize : 15
%     bsize            : 2
% Output:
%     weightA    : 1 x 1 x 1 x B
%     weightD    : 1 x 1 x 1 x B
%     CCT        : 1 x 1 x 1 x B
%     Fweights   : 1 x 1 x 12 x B
%     b          : 2 x nbatch
%-------------------------------- CNN -------------------------------------
[ x,y ] = fcn_multipleDecoders( nfilters,images,nclass,true );
%% ---------------------  Fully connected layers --------------------------
% FC1
fc1 = vl_nnconv(y, 'size', [4 4 512 512]) ;  
fc1 = vl_nnbnorm(fc1) ;
fc1 = vl_nnrelu(fc1) ;
% FC2
fc2 = vl_nnconv(fc1, 'size', [1 1 512 512]) ;
fc2 = vl_nnbnorm(fc2) ;
fc2 = vl_nnrelu(fc2) ;
% FC3
dims = LightVectorSize + bSize ; 
prediction = vl_nnconv(fc2, 'size', [1 1 512 dims]);
prediction.name = 'prediction';
%% --------------------- Illumination parameters --------------------------
lightingparameters = prediction(:,:,1:LightVectorSize,:);
%% ------------------------ camera parameters -----------------------------
nbatch = size(prediction,4);
b = reshape(prediction(:,:,LightVectorSize+1:LightVectorSize+bSize,:),[bSize nbatch]);  
%% ------------------------- Predicted Maps -------------------------------
fmel     = x(:,:,1,:);
fblood   = x(:,:,2,:);
Shading  = x(:,:,3,:);
specmask = x(:,:,4,:);


end

