function y = fillMiss(input,output)

x = csvread(input);
y = fillmissing(x,'pchip');


dlmwrite(output, y , 'delimiter', ',', 'precision',11);


end