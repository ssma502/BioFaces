function [mu,PC,EV]= CameraSensitivityPCA(cmf)

X = zeros(99,28);
Y = zeros(99,28);
redS =cmf.rgbCMF{1,1};
greenS= cmf.rgbCMF{1,2};
blueS =cmf.rgbCMF{1,3};
for i=1:28

    Y(1:33,i)=redS(:,i)./sum(redS(:,i));
    Y(34:66,i)=greenS(:,i)./sum(greenS(:,i));
    Y(67:99,i)=blueS(:,i)./sum(blueS(:,i));

end
[PC,~,EV,~,explained,mu] = pca(Y');
PC = single(PC(:,1:2)*diag(sqrt(EV(1:2))));  
mu = single(mu');
EV = single(EV(1:2));

end
