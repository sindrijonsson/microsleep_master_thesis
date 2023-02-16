out = load("Otq3.mat");
tmp = out.Data;

y = tmp.eeg_O1;

fs = tmp.fs;
win = 1:200;
shift = 40;
secShift = 1*(shift/fs);


len = tmp.num_Labels;
freq = 0:0.01:30;
numWins = (len / shift) - ceil((fs / shift) - 1);
yy = zeros(length(freq), numWins);
tt = 0.5:shift/fs:(len/fs)-0.5;

hWin = hann(length(win));

i = 1;
while max(win) <= length(y)
    tmpY = y(win).*hWin;
    [H,F] = pburg(tmpY, 16, freq, tmp.fs);
    yy(:,i) = pow2db(H);
    i = i + 1;
    win = win + shift;
end


%%

imagesc(tt, f, yy); 
axis xy
colormap jet
colorbar
caxis([-20 25])
xlabel("Time [s]")
ylabel("Frequenzy [Hz]")