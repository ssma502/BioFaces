function inputs = getBatch(imdb, batch)
realimages  = imdb.images.data(:,:,:, batch);
%actualfmel = imdb.images.fmel(:,:,:, batch);
%actualfblood = imdb.images.fblood(:,:,:, batch);
actualshading = max(0,imdb.images.shading(:,:,:, batch));
%actualshading = imdb.images.shading(:,:,:, batch);
%actualspecular = imdb.images.specular(:,:,:, batch);
actualmasks    = imdb.images.mask(:,:,:, batch);

images = (zeros(64,64,3,length(batch),'single'));
im = zeros(64,64,3,'single');
mu = imdb.averageImage;

for i = 1: length(batch)
for j=1:3
      im(:,:,j) = 255.*((realimages(:,:,j,i)).^2.2);
% im(:,:,j) = 255.*(realimages(:,:,j,i));
%     im(:,:,j) = (realimages(:,:,j,i));
     im(:,:,j)=im(:,:,j)-mu(j);
end
images(:,:,:,i) = im;
end
  inputs = {'images', images,'actualshading',actualshading,'actualmasks',actualmasks};

%   inputs = {'images', images,'actualmasks',actualmasks};

% inputs = {'images', images,'actualfmel',actualfmel,'actualfblood',actualfblood, ...
 %   'actualshading',actualshading,'actualspecular',actualspecular,'actualmasks',actualmasks};
 


