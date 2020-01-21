function res = SER(x,y)


res = 10*log10(sum(x.^2)/sum((x-y).^2));

end