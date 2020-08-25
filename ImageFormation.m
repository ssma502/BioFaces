function [ rawAppearance,diffuseAlbedo ] = ImageFormation (R_total, Sr,Sg,Sb,e,Specularities,Shading)
%Inputs:,
%     R_total       : H X W X 33 X nbatch
%     Shading       : H X W X 1 X nbatch
%     Specularities : H X W X 1 X nbatch
%     Sr,Sg,Sb      : 1 x 1 x 33 x nbatch
%     e             : 1 x 1 x 33 x nbatch
% Output:
%     rgbim : H x W x 1 x nbatch
%---------------------------Image Formation -------------------------------
spectraRef = R_total.*e ;  
%--------------------------------------------------------------------------
rChannel = sum((spectraRef.*Sr),3);  
gChannel = sum((spectraRef.*Sg),3);  
bChannel = sum((spectraRef.*Sb),3);  

diffuseAlbedo = cat(3,rChannel,gChannel,bChannel);  % H x W x 3 x nbatch

%---------------------------Shaded Diffuse --------------------------------

ShadedDiffuse = diffuseAlbedo.*Shading;  %ShadedDiffuse.name ='ShadedDiffuse';

%---------------------------Raw appearance --------------------------------
rawAppearance = ShadedDiffuse + Specularities ;  

end