function [ R_total ] = BiotoSpectralRef(fmel,fblood,Newskincolour)
% Inputs:
%     fmel             : H x W x 1 x B
%     fblood           : H x W x 1 x B
%     Newskincolour    : 256 x 256 x 33 x B
%  Output:
%     R_total          : H x W x 33 x B
%%
BiophysicalMaps = cat(3, fblood,fmel);
BiophysicalMaps = permute(BiophysicalMaps,[3 1 2 4]); % 2 x H x W x B


R_total  = vl_nnbilinearsampler(Newskincolour,BiophysicalMaps); 
end