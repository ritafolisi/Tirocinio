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

    % seleziono features e labels per iris normale
    %features = T{:,1:end-1};
    %labels = T{:,end};

    % seleziono features e labels per iris setosa/viriginica etc
    features = T{:,2:3};
    labels = T{:,1};
    
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

% Number of Fold
k_fold = 3; % Edit here for adjusting the fold value for cross validation.

cv = cvpartition(size(dataset,1), 'kfold', k_fold);
acc = 0; answer = [];

for count = 1:k_fold    % questa crossvalidation sarebbe da togliere, 
                        % considerato che questa sarà la parte della
                        % predict e calcolo dello score
    
    trainMatrix = dataset(training(cv,count), : );
	testMatrix = dataset(test(cv,count), : );

    acc = score (trainMatrix, testMatrix);
    
    answer = [answer;acc];
end

h = sprintf('Acc : %.2f (%.2f)',mean(answer),std(answer));
disp(h)
end