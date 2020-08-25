function [ e ] = illuminationModel(weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm)
% Inputs:
%     weightA    : 1 x 1 x 1 x B
%     weightD    : 1 x 1 x 1 x B
%     CCT        : 1 x 1 x 1 x B
%     Ftype      : 1 x 1 x 12 x B
%     illumA     : 1 x 1 x 33 x B
%     illumDNorm : 1 x 1 x 33 x 22
%     illumFNorm : 1 x 1 x 33 x 12
%  Output:
%     e          : 1 x 1 x 33 x B
%% ------------------  create the illumination model ----------------------
% illumination A:
illuminantA = illumA.*weightA; 

% illumination D:
illumDlayer = Layer.fromFunction(@vl_nnillumD);
illD   = illumDlayer(CCT,illumDNorm);
illuminantD = illD.*weightD; 

% illumination F:
illumFNorm = permute(illumFNorm,[1 3 4 2]); %permute to 1 x 33 x 12 x 1
illuminantF = illumFNorm.*Fweights; % 1 x 33 x 12 x B
illuminantF = sum(illuminantF,3); % 1 x 33 x 1 x B
illuminantF = permute(illuminantF,[1 3 2 4]);
% illuminantF.name ='illuminantF';

e = illuminantA + illuminantD +illuminantF ;
esums = sum(e,3);
e = e./esums; 

if isa(weightA, 'Layer')
 illuminantA.name = 'illuminantA';
 illuminantD.name='illuminantD';
 illuminantF.name ='illuminantF';
end
% e.name = 'e';
end

