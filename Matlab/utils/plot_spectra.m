function ax = plot_spectra(tyy, freq, pyy, ax)
    
    if nargin < 4; ax = gca; end

    imagesc(tyy, freq, pyy);
    axis xy
    colormap jet
    xlabel("Time [s]")
    ylabel("Freq. [Hz]");
    ax = gca;

end