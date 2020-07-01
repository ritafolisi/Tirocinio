function [nomefile, accuratezza] = esperimento_main()
    %read data
    [features,labels,no_types,divisions]=read_iris_data();

    %normalise each feature

    for i = 1 : size(features,2)
        features(:,i)= (features(:,i)-min(features(:,i)))/(max(features(:,i))- min(features(:,i)));
    end

    %feed data to this helper func
    accuratezza = run_cluster_fc_means( features,labels,no_types,divisions);
    disp(accuratezza)
    nomefile = "esperimento_deserialize.m";

end

function [features,labels,no_types,divisions]=read_iris_data() 
    %read iris data set
    fprintf("Working on IRIS data set\n");
 
    %read dataset into table
    data=readtable("iris-setosa.csv");
    
    %retrieve data into arrays.
    features=data{:,["sepal_length", "sepal_width"]}; 
    labels= data{:,"species"};
    divisions=[0, 50, 150];
    [labels,no_types]=convert_labels(labels);
end


function [labels,no_types] = convert_labels(labels_in_data)
    no_points=size(labels_in_data,1);
    labels=zeros(no_points,1);
    label_count=1;
    label_map=containers.Map;
    for i = 1:size(labels_in_data,1)
        label=char(labels_in_data(i));
        if isKey(label_map,label)~= 1
            label_map(label)=label_count;
            label_count=label_count+1;
        end
        labels(i)=label_map(label);
    end
    no_types=label_count-1;
end
