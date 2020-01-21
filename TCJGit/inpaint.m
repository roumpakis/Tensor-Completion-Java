function y = inpaint(input,output)

x = csvread(input);
y = inpaintn(x);


dlmwrite(output, y , 'delimiter', ',', 'precision',11);


end