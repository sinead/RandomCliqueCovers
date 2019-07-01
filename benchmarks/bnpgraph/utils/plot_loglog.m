function [h, centerbins, out] = plot_loglog(x, y, linespec, pas)

if nargin<3
    linespec = '*';
end
if nargin<4
    pas = 1;
end

edgebins = 2.^(0:pas:12);
sizebins = edgebins;
sizebins = edgebins(2:end) - edgebins(1:end-1);

sizebins(end+1) = 1;
centerbins = edgebins;
% centerbins(2:end) = (edgebins(2:end) + edgebins(1:end-1))/2;
% centerbins(end+1) = edgebins(end);
[counts, bins] = histc(x, edgebins);
out = zeros(size(counts));
for i=1:length(counts)
    if counts(i)>0
        out(i) = mean(y(bins==i));
    else
        out(i) = NaN;
    end
end
h = loglog(centerbins, out, linespec);
