function [ x,y ] = fcn_multipleDecoders( nfilters,x,nclass,doubleconv )
%FCN Build an autonn fully convolutional network for image to image tasks
% Includes skip connections from each encoder layer to corresponding
% decoder layer
%   Inputs:
%       nfilters - vector of filter sizes, 1:length-1 are used in the
%       contractive part of the network, the final value is the number of
%       filters applied at the lowest resolution before upsampling begins
%       x - input feature map as autonn layer
%       nclass - number of channels in output feature map
% E.g. nfilters = [32 32 64 96 128 256];
%
%   Outputs:
%       x - image to image prediction
%       y - output of lowest resolution - can be fed into fully connected
%       layers

nlayers = length(nfilters);

% Encoder
for i=1:nlayers-1
    if i==1
        x = vl_nnconv(x, 'size', [3 3 3 nfilters(i)], 'pad', 1);
    else
        x = vl_nnconv(x, 'size', [3 3 nfilters(i-1) nfilters(i)], 'pad', 1);
    end
    x = vl_nnbnorm(x);
    if doubleconv
        x = vl_nnrelu(x);
        x = vl_nnconv(x, 'size', [3 3 nfilters(i) nfilters(i)], 'pad', 1);
        x = vl_nnbnorm(x);
        x = vl_nnrelu(x);
        x = vl_nnconv(x, 'size', [3 3 nfilters(i) nfilters(i)], 'pad', 1);
        x = vl_nnbnorm(x);
    end
    x_skip{i} = vl_nnrelu(x);
    x = vl_nnpool(x_skip{i}, 2, 'stride', 2) ;
end

x = vl_nnconv(x, 'size', [3 3 nfilters(nlayers-1) nfilters(nlayers)], 'pad', 1);
x = vl_nnbnorm(x);
y = vl_nnrelu(x);
if doubleconv
    x = vl_nnconv(y, 'size', [3 3 nfilters(nlayers) nfilters(nlayers)], 'pad', 1);
    x = vl_nnbnorm(x);
    y = vl_nnrelu(x);
    x = vl_nnconv(y, 'size', [3 3 nfilters(nlayers) nfilters(nlayers)], 'pad', 1);
    x = vl_nnbnorm(x);
    y = vl_nnrelu(x);
end



% Decoders
vl_nnupsample = Layer.fromFunction(@autonn_upSample);

for c=1:nclass
    
    for i=nlayers-1:-1:1
        if i==nlayers-1
            x = vl_nnupsample(y);
        else
            x = vl_nnupsample(x);
        end
        x = cat(3,x,x_skip{i});
        x = vl_nnconv(x, 'size', [3 3 nfilters(i)+nfilters(i+1) nfilters(i)], 'pad', 1);
        x = vl_nnbnorm(x);
        x = vl_nnrelu(x);
        
        if doubleconv
            x = vl_nnconv(x, 'size', [3 3 nfilters(i) nfilters(i)], 'pad', 1);
            x = vl_nnbnorm(x);
            x = vl_nnrelu(x);
            
            x = vl_nnconv(x, 'size', [3 3 nfilters(i) nfilters(i)], 'pad', 1);
            x = vl_nnbnorm(x);
            x = vl_nnrelu(x);
        end
    end
    % Final predictions
    x = vl_nnconv(x, 'size', [3 3 nfilters(1) 1], 'pad', 1);
    if c==1
        z = x;
    else
        z = cat(3,z,x);
    end
    
end
x=z;

end
