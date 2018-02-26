clear all
close all
clc

% Loading the training data..................................................

%Ntrain=xlsread('ACTREC3D_27');
Ntrain=xlsread('BERK7525_27');

[NTD,~] = size(Ntrain);
y = Ntrain(:, end);

% Initialize the Algorithm Parameters.....................................
inp = 391;              % No. of input neurons
hid = 20;            % No. of hidden neurons
out =11;            % No. of Output Neurons
lam = 0.01;         % Learning rate
epo = 4000;          % epoch value

% Dividing our data into random 80%-20% divisions for use in training and validation
rNTD = randperm(NTD, NTD);      % getting a random permutation for the samples
T = mod(NTD, 10);
Te = NTD - T;
T_tr = 0.8*Te;                  % Number of samples for training          
T_va = NTD - T_tr;              % Number of samples for validation
TR = rNTD(1:T_tr);
VA = rNTD(T_tr+1:end);

% Added coded vectors at the end of samples
%Ntrain = Ntrain(:, 1:inp);
%Ntrain = [Ntrain Y];

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights
er = zeros(out, 1);
NTD = T_tr;                          % Number of training data

% Train the network.......................................................
e = zeros(epo,1);
m = zeros(epo,1);
z = zeros(epo,1);
e_validation=zeros(epo,1);
for ep = 1 : epo
    sumerr = 0;
    miscla = 0;
    for sa = 1 : T_tr
        xx = Ntrain(TR(sa),1:inp)';                         % Current Sample
        tt = Ntrain(TR(sa),inp+1:end)';                     % Current Target
        Yh = 1./(1+exp(-Wi*xx));                        % Hidden output
        Yo = Wo*Yh;                                     % Predicted output

        er = tt - Yo;                                 % Error
        Wo = Wo + lam * (er * Yh');                   % update rule for output weight
        Wi = Wi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';   % update for input weight
        sumerr = sumerr + sum(er.^2);                 % LS error

        ca = find(tt==1);                               % actual class
        [~,cp] = max(Yo);                               % Predicted class
        if ca~=cp 
            miscla = miscla + 1;
        end
        % er = zeros(out,1);                              % For MLS error 
        % for ac = 1 : out
        %     if Yo(ac) * tt(ac) <1 
        %         er(ac) = tt(ac) - Yo(ac);
        %     end    
        % end
        
       % Wo = Wo + lam * (er * Yh');                     % update rule for output weight
       % Wi = Wi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';     % update for input weight
       % sumerr = sumerr + sum(er.^2);

    end
    e(ep) = sqrt(sumerr/T_tr);
    m(ep) = miscla;
    z(ep) = ep;



rmstra=0;
% Validate the network.....................................................
conftra = zeros(out,out);
res_tra = zeros(T_va, 2);
for sa = 1:T_va
        xx = Ntrain(VA(sa),1:inp)';     % Current Sample
        tt = Ntrain(VA(sa),inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        conftra(ca,cp) = conftra(ca,cp) + 1;
        res_tra(sa,:) = [ca cp];
        er= tt-Yo;
        %sumerr = sumerr + sum(er.^2);   
        rmstra = rmstra + sum(er.^2);
end
e_validation(ep)=sqrt(rmstra/T_va);

end

disp('Training error');
disp(e(epo));
disp('Validation Error');
disp(e_validation(epo));
% Plotting RMS Error against Number of epochs
figure(1);
title('Convergence against no. of epochs');
plot(z(500:end), e(500:end), 'k', 'linewidth', 4),hold on;
plot(z(500:end), e_validation(500:end), 'r', 'linewidth', 4);
xlabel('epo');
ylabel('Sum of squared error');


% Finding the Geometric mean Accuracy for the validation part
stor = zeros(out, 1);
for c = 1:out
    q = conftra(c, c);
    n = sum(conftra(c, 1:end));
    stor(c, 1) = q*100/n;
end
ng = nthroot(prod(stor(:)), out)
disp('Geometric Accuracy')
disp(ng);




% Test the network.........................................................
%NFeature=xlsread('ACTREC3D_test_s27');
NFeature=xlsread('BERK_test_s27');

[NTD,~]=size(NFeature);
%conftes = zeros(out,out);
res_tes = zeros(NTD,1);
result = zeros(out, 1);
for sa = 1: NTD
        xx = NFeature(sa,1:inp)';       % Current Sample
        %ca = NFeature(sa,end);         % Actual class
        Yh = 1./(1+exp(-Wi*xx));        % Hidden output
        Yo = Wo*Yh;                     % Predicted output
        [~,cp] = max(Yo);               % Predicted class
        result(cp, 1)= result(cp,1)+ 1;
      %  conftes(ca,cp) = conftes(ca,cp) + 1;
      res_tes(sa,:) = [cp];
end


filename='mlp_ls_berk.xlsx';
xlswrite(filename,res_tes);


%disp(conftra)
qii = 0;
for c = 1 : size(conftra,1)
    qii = qii + conftra(c,c);
end

overall_accuracy = (100*qii)/sum(conftra(:));
disp('Overall Accurracy');
disp(overall_accuracy);

Ni = 0;
QiibyNi = zeros(size(conftra,1), 1);
for i = 1 : size(conftra,1)
    Ni = Ni + sum(conftra(i, :), 2);
    QiibyNi(i) = (conftra(i,i) / Ni);
end

 %Average Accuracy
      val=0;
      for i=1:out
         val=val+conftra(i,i)/sum(conftra(i,:));
      end
      na=100/out*(val);
      disp('Average Accuracy');
      disp(na);