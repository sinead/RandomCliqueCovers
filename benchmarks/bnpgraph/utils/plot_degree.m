function [h2, centerbins, freq] = plot_degree(G, linespec, pas)

if nargin<2
    linespec = '*';
end
if nargin<3
    pas = 1;
end


deg = full(sum(G));
% [f, x] = hist(deg, min(deg):max(deg));
% % h=figure;
% f_norm = f/size(G,1);
% loglog(x, f_norm, '*')
% xlabel('Degree', 'fontsize', 16)
% ylabel('Distribution', 'fontsize', 16)


% Uses logarithmic binning to get a less noisy estimate of the
% pdf of the degree distribution

edgebins = 2.^(0:pas:12);
sizebins = edgebins;
sizebins = edgebins(2:end) - edgebins(1:end-1);

sizebins(end+1) = 1;
centerbins = edgebins;
% centerbins(2:end) = (edgebins(2:end) + edgebins(1:end-1))/2;
% centerbins(end+1) = edgebins(end);
counts = histc(deg, edgebins);
freq = counts./sizebins/size(G, 1);
h2=loglog(centerbins, freq, linespec);
xlabel('Degree', 'fontsize', 16)
ylabel('Distribution', 'fontsize', 16)


% ctrbins = cumsum(sizebins);
% deltabin = [ones(1, 9), 3, 5, 7, 10*ones(1, 7), 55, 100*ones(1, 9)];
% size(bins)
% size(deltabin)
% [f, x] = hist(deg, bins);
% h2=figure;
% size(f)
% f_norm = f./deltabin/size(G,1);
% loglog(x, f_norm, '*')
% xlabel('Degree', 'fontsize', 16)
% ylabel('Distribution', 'fontsize', 16)


% % Uses some sort of logarithmic binning to get a less noisy estimate of the
% % pdf of the degree distribution
% 
% bins = [1:10, 15, 20:10:100, 200:100:1000];
% deltabin = [ones(1, 9), 3, 5, 7, 10*ones(1, 7), 55, 100*ones(1, 9)];
% size(bins)
% size(deltabin)
% [f, x] = hist(deg, bins);
% h2=figure;
% size(f)
% f_norm = f./deltabin/size(G,1);
% loglog(x, f_norm, '*')
% xlabel('Degree', 'fontsize', 16)
% ylabel('Distribution', 'fontsize', 16)
