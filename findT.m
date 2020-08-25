function [T_RAW2XYZ] = findT(Tmatrix,BGrid)
% Inputs:
%     Tmatrix          : 128 x 128 x 9 
%     BGrid            : 2 x B
%  Output:
%     T_RAW2RGB        : 1 x 1 x 9 x B
%%
 T_RAW2XYZ = vl_nnbilinearsampler(Tmatrix,BGrid);
 %T_RAW2RGB.name ='T_RAW2RGB';
end

