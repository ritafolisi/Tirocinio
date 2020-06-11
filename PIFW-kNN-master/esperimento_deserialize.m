function esperimento_deserialize(filename)

global b f k membership W 

    fileID = fopen('esperimento_W.bin','r');
    b1 = fread(fileID);
    W = hlp_deserialize(b1);
    fclose(fileID);

    fileID = fopen('esperimento_membership.bin','r');
    b2 = fread(fileID);
    membership = hlp_deserialize(b2);
    fclose(fileID);

    fileID = fopen('esperimento_bestk.bin','r');
    b3 = fread(fileID);
    k = hlp_deserialize(b3);
    fclose(fileID);
    
    % leggo file csv 
    T = readtable(filename);

    % seleziono features e labels
    features = T{:,1:end-1};
    labels = T{:,end};
    
    Temp_Features = features;
    Temp_Labels = labels;

% Normalizing Features
max_x = max(Temp_Features,[],1);
min_x = min(Temp_Features,[],1);
Temp_Features = (Temp_Features-repmat(min_x,[size(Temp_Features,1),1]))./repmat(max_x-min_x,[size(Temp_Features,1),1]);

[~, ~, Temp_Labels] = unique(Temp_Labels);

dataset = [Temp_Features,Temp_Labels];

% Number of Labels
b = max(Temp_Labels);

% Number of Features
f = size(Temp_Features,2);

[m,n] = size(dataset) ;
P = 0.70 ;
idx = randperm(m)  ;
trainMatrix = dataset(idx(1:round(P*m)),:) ; 
testMatrix = dataset(idx(round(P*m)+1:end),:) ;

matched = 0; unmatched = 0;

for i = 1:size(testMatrix,1)
		lab = fuzzy_knn(trainMatrix,testMatrix(i,:));
		if lab == testMatrix(i,size(testMatrix,2))
			matched = matched + 1;
		else
			unmatched = unmatched+1;
		end
end
    
acc = matched/(matched+unmatched);
acc = acc*100;

disp(acc)
end