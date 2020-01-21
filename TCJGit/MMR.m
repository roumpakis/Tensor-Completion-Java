function res = MMR(x,y,N)


res = ((1/N) * sum(abs(x-y)))/((1/N) * sum(abs(x)));

end