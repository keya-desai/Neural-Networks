% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
Ntrain=xlsread('fin_27');
[NTD,dim] = size(Ntrain);



TR=floor(3/4*(NTD));

% Initialize the Algorithm Parameters.....................................
inp = 10;          % No. of input neurons
hid = 10;        % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 1.e-07;       % Learning rate
epo = 1000;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights
e = zeros(epo,1);
m = zeros(epo,1);
z = zeros(epo,1);
e_val=zeros(epo,1);
% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    
    for sa = 1 : TR
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
       % Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yh = 2./(1+exp(-2*Wi*xx))-1;
        %Yh=(Wi*xx);
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
        Wi = Wi + DWi;
        Wo = Wo + DWo;
    end
     %e(ep) = sqrt(sumerr/TR);
   
    %z(ep) = ep;
    
end

%if h==max_hidden
disp('Training error');
disp(sqrt(sumerr/TR));

%e_hidden(h)=sqrt(sumerr/TR);
%z_hidden(h)=h;

% Validate the network.....................................................
rmstra = zeros(out,1);
%res_tra = zeros(NTD,2);
for sa = TR+1: NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        %Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yh = 2./(1+exp(-2*Wi*xx))-1;
       
        Yo = Wo*Yh;                 % Predicted output
        rmstra = rmstra + (tt-Yo).^2;
       
end


disp('validation error');
disp(sqrt(rmstra/(NTD-TR)));






% Test the network.........................................................
NFeature=xlsread('fin_test_s27');
[NTD,~]=size(NFeature);
res=zeros(NTD,1);
for sa = 1: NTD
        xx = NFeature(sa,:)';   % Current Sample
        
        %Yh = 1./(1+exp(-Wi*xx));% Hidden output
       
        Yh = 2./(1+exp(-2*Wi*xx))-1;
       
        Yo = Wo*Yh;
       res(sa,1)=Yo;
        
end
filename='mlp_approx_fin.xlsx';
xlswrite(filename,res);


