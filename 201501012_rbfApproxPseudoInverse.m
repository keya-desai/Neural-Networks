% RBF Approximation - pseudo inverse matrix method
close all;
clc;

% note : the inputs aren't normalised in case of SI.xlsx dataset and the number of epochs,
% hidden neurons, learning rate, number of samples for training and validation is different
% as mentioned in result table

% Load training data----------------------------------------
Ntrain = xlsread('SI_27.xlsx');
[rows, dim] = size(Ntrain);

% Initialize the algorithm parameters----------------------
inp = 10;
hid = 15;
output = 1;
LR= 1.e-05;
epochs=1000;
hidOutp= zeros(rows,hid);
weightMat = zeros(hid, 1);

% Initialize the centres-------------------------------------
R = randperm(rows);
C = zeros(hid, dim-1);
for k=1:hid
    C(k,:) = Ntrain(R(k),1:dim-1);
end

% spread---------------------------------------------------------
dmax=0;
for i=1:k  
    for j=1:k
        dist = abs(norm(C(i,:)-C(j,:)));
        if(dmax<dist)
            dmax = dist;
        end
    end
end
Spread = dmax/sqrt(hid)


% Training the network
for epoch=1:epochs
    errSum=0;
    for i=1:750
        for j=1:hid
            hidOutp(i,j) = exp(-(norm(Ntrain(i,1:dim-1)-C(j,:)).^2)/(2*Spread*Spread));
        end
    end
    weightMat = pinv(hidOutp)*Ntrain(:,dim);
end

% Validate the network
rmstra = zeros(output,1);
res_tra = zeros(rows,2);
for i = 751:1000
    for j=1:hid
        hidOutp(i,j) = exp(-(norm(Ntrain(i,1:dim-1)-C(j,:)).^2)/(2*Spread*Spread));
    end
end
tt = Ntrain(:,dim);
yy = hidOutp*weightMat;
rmstra = rmstra + (tt-yy).^2;
disp(sqrt(rmstra/rows))