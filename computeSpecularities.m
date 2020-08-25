function [ Specularities ] = computeSpecularities(specmask,lightcolour)
% Inputs:
%     specmask          : H x W x 1 x B
%     lightcolour      : 1 x 1 x 3 x B
%  Output:
%     Specularities    : H x W x 1 x B
%%
Specularities = specmask.*lightcolour;  
end

