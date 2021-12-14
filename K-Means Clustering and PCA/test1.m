clc
clear all;
close all;
nSample=1000;
M = 4;                  % modulation index for psk
hpsk = comm.PSKModulator('ModulationOrder',M,...
    'BitInput',false,...
    'PhaseOffset',0);   % M-psk modulator
snr_dB = [-9.6,-7.8,-9,-4.1,-7.9,-3.2,-8,-10,-9.5,-10.2,-4.5,-8.3,-3.6,-9.1,-7.9,-8.5,-3,-4.4,-10.1,-8.2]; % SNR in decibels
label = [1,1,1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1];
%energySum = snr_dB';
energySum=zeros(length(snr_dB),1);
for k=1:1000
    eSUM=zeros(length(snr_dB),1);
for juser=1:length(snr_dB)            
    infoSignal = randi(M,nSample,1)-1;
          txSignal = step(hpsk, infoSignal);
             rxSignal = awgn(txSignal,snr_dB(juser));
             snr = 10^(snr_dB(juser)./20);
             nvar = 1/snr ;
             N = length(rxSignal);
             noise = sqrt(0.1)*(randn(1,N)+juser*randn(1,N));
             eSUM(juser) = (1/nSample).*sum(abs(rxSignal).^2);

end
energySum = [energySum,eSUM];
end
energymean=mean(energySum,2);
energySum = [snr_dB',energymean];
save('data.mat', 'energySum') 