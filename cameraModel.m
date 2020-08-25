function [Sr,Sg,Sb] = cameraModel(mu,PC,b,wavelength)
% Inputs:
%     mu         : 1 x 1 x 1 x B
%     PC         : 1 x 1 x 1 x B
%     b          : 1 x 1 x 2 x B
%     wavelength : 33

% Outputs:
%     Sr,Sg,Sb   : 1 x 1 x 33 x B
nbatch = size(b,2);
%% PCA model
S = PC*b + mu; % 99 x nbatch
S =  vl_nnrelu(S) ;   
% S.name='S'; % Clamp negative values to zero: positive 99 x nbatch
%% split up S into Sr, Sg, Sb 
Sr = reshape(S(1:wavelength,:),[1 1 wavelength nbatch]);                  
Sg = reshape(S(wavelength+1:wavelength*2,:),[1 1 wavelength nbatch]);     
Sb = reshape(S((wavelength*2)+1:wavelength*3,:),[1 1 wavelength nbatch]);

end

