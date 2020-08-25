function [lightcolour] = computelightcolour(e,Sr,Sg,Sb)
% Inputs:
%     Sr,Sg,Sb         : 1 x 1 x 33 x B
%     e                : 1 x 1 x 33 x B
%  Output:
%     lightcolour        : 1 x 1 x 3 x B
%% ------------------------ lightcolour -----------------------------------
lightcolour = [sum(Sr.*e,3) sum(Sg.*e,3) sum(Sb.*e,3)];
lightcolour = reshape(lightcolour,[1 1 3 size(lightcolour,4)]);
 
end