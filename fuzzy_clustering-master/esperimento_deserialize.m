
fileID = fopen('esperimento_u_mat.bin','r');
b1 = fread(fileID);
u_mat = hlp_deserialize(b1);
fclose(fileID);

