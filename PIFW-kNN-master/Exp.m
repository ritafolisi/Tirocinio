    %importo csv
T=readtable('iris.csv');

    % seleziono features e labels
features = T{:,1:end-1};
labels = T{:,end};

pifwknn(features, labels)

