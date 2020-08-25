function [sRGBim ] = fromRawTosRGB(imWB,T_RAW2XYZ)
% Inputs:
%     imWB: H X W X 3 X B
%     Tmatrix :  128 X 128 X 9 
% Output:
%     sRGBim : H x W x 3 x B
%%
   Ix = T_RAW2XYZ(1,1,1,:).*imWB(:,:,1,:) + T_RAW2XYZ(1,1,4,:).*imWB(:,:,2,:) + T_RAW2XYZ(1,1,7,:).*imWB(:,:,3,:); 
   Iy = T_RAW2XYZ(1,1,2,:).*imWB(:,:,1,:) + T_RAW2XYZ(1,1,5,:).*imWB(:,:,2,:) + T_RAW2XYZ(1,1,8,:).*imWB(:,:,3,:); 
   Iz = T_RAW2XYZ(1,1,3,:).*imWB(:,:,1,:) + T_RAW2XYZ(1,1,6,:).*imWB(:,:,2,:) + T_RAW2XYZ(1,1,9,:).*imWB(:,:,3,:); 
 
  Ixyz = cat(3,Ix,Iy,Iz);
  
  Txyzrgb = single([3.2406, -1.5372, -0.4986; -0.9689, 1.8758, 0.0415; 0.0557, -0.2040, 1.057]);

  if isa(imWB, 'Layer')
      Txyzrgb = Param('value',Txyzrgb,'learningRate',0); Txyzrgb.name='Txyzrgb';
  end 
  
 R = Txyzrgb(1).*Ixyz(:,:,1,:) + Txyzrgb(4).*Ixyz(:,:,2,:) + Txyzrgb(7).*Ixyz(:,:,3,:);  % R
 G = Txyzrgb(2).*Ixyz(:,:,1,:) + Txyzrgb(5).*Ixyz(:,:,2,:) + Txyzrgb(8).*Ixyz(:,:,3,:);  % G
 B = Txyzrgb(3).*Ixyz(:,:,1,:) + Txyzrgb(6).*Ixyz(:,:,2,:) + Txyzrgb(9).*Ixyz(:,:,3,:);  % B

 sRGBim = cat(3,R,G,B);
 sRGBim = vl_nnrelu(sRGBim);

end

