function [weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,Shading,specmask] = scalingNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize)
% Inputs/Output:
%     weightA  : 1 x 1 x 1 x B
%     weightD  : 1 x 1 x 1 x B
%     CCT      : 1 x 1 x 1 x B
%     Fweights : 1 x 1 x 12 x B
%     b        : 1 x 1 x 2 x B
%     fmel     : 224 x 224 x 1 x B
%     fblood   : 224 x 224 x 1 x B
%     Shading  : 224 x 224 x 1 x B
%     specmask : 224 x 224 x 1 x B
%     bSize    : 2
% Output:
% Scaled inputs
nbatch = size(lightingparameters,4);

lightingweights = vl_nnsoftmax(lightingparameters(:,:,1:14,:));
weightA  = lightingweights(:,:,1,:);
weightD  = lightingweights(:,:,2,:);
Fweights = lightingweights(:,:,3:14,:);
CCT      =  lightingparameters(:,:,15,:); 
CCT      = ((22 - 1)./ (1 + exp(-CCT))) + 1;  
%%
b = 6.*(vl_nnsigmoid(b))-3;  
BGrid = reshape(b,[bSize 1 1 nbatch]); %  2 x 1 x 1 x B
BGrid = BGrid./3;     % [ -1 , +1 ] 
%%
fmel = vl_nnsigmoid(fmel).*2-1; 
fblood = vl_nnsigmoid(fblood).*2-1; 
Shading = exp(Shading);   
specmask= exp(specmask);   

end

