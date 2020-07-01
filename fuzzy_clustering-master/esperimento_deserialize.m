function[] = esperimento_deserialize(filename, col_feature1, col_feature2, col_features, divisions)
    fileID = fopen('esperimento_m_val.bin','r');
    b1 = fread(fileID);
    m_val = hlp_deserialize(b1);
    disp(m_val)
    fclose(fileID);

    fileID = fopen('esperimento_centers.bin','r');
    b2 = fread(fileID);
    new_centers = hlp_deserialize(b2);
    fclose(fileID);

    fileID = fopen('esperimento_no_clusters.bin','r');
    b3 = fread(fileID);
    no_clusters = hlp_deserialize(b3);
    fclose(fileID);

    [features,output_labels]=read_iris_data(filename, col_feature1, col_feature2, col_features);
    
    for i = 1 : size(features,2)
        features(:,i)= (features(:,i)-min(features(:,i)))/(max(features(:,i))- min(features(:,i)));
    end
    
    %cv = cvpartition(size(features,1), 'kfold', 5);
    %for count = 1:5
    %    trainMatrix = features(training(cv,count), : );
    %    testMatrix = dataset(test(cv,count), : );
	
    no_points=size(features,1);
    dist=zeros(no_points,no_clusters);

    for j = 1 : no_clusters
        t1=features-new_centers(j,:);
        t1=t1.^2;
        t2= sum (t1,2);
        %sum along 2nd index, that is get n x 1 matrix
        t2 = sqrt(t2);
        dist(:,j)= t2;
    end

    u_mat=calculate_fcm_memberships(features,new_centers,dist,m_val);
    
    %hard partition
    labels=zeros(no_points,1);
    for i = 1 : no_points
        max_membership=0;
        for j = 1 : no_clusters 
            if ( u_mat (i,j) > max_membership)
                max_membership=u_mat(i,j);
                
                labels(i) = j;
            end
        end
    end

    estimate_error(new_centers, labels, output_labels, no_points, divisions);    
    %err=immse(output_labels, labels);
    %disp(err);
    %disp(labels);    
end


function estimate_error(centers,output_labels,labels,no_points, divisions)
    no_clusters=size(divisions,2)-1;
    % fprintf("clusters is ");
    % disp(no_clusters);
    label_map=containers.Map(1,1);
    %disp(label_map);
    % disp (divisions);
    % fprintf("What is size : %d\n",size(divisions,2));
    for j=1:size(divisions,2)-1
    %     sprintf("J is %d\n;");
        label_map(mode(output_labels((divisions(j)+1):divisions(j+1))))=j;
    end
     %disp(label_map.keys());
     %disp(label_map.values());

    %calculate errors
    if size(keys(label_map),2)==no_clusters
        error_count=0;
        for i = 1:no_points 
            if label_map(output_labels(i)) ~= labels(i) 
               %          disp([i , label_map(output_labels(i)), labels(i)]);
              %           disp (features(i,:));
                error_count=error_count+1;
            end
        end
        fprintf("Error count = %d & Success Rate = %f \n",error_count,(no_points-error_count)* 100/no_points );
    else
        %fprintf("Mode method of mapping fails : Likely too many errors!\n")
    end

end   
function u_mat = calculate_fcm_memberships(points,new_centers,dist, m)

    no_points=size(points,1);
    no_clusters=size(new_centers,1);
    
    u_mat=zeros(no_points,no_clusters);
    
    for i = 1: no_points
        %calculate memberships for jth cluster
        for j = 1:no_clusters
            if dist(i,j)==0 %point is center of cluster j, therefore full membership
                u_mat(i,j)=1;
                continue;
            end
            denom=0;
            done=0;
            for k = 1: no_clusters
                if (dist(i,k)==0) %it is a center of different cluster k, 0 membership to jth cluster
                    u_mat(i,j)=0;
                    done=1;
                    break;
                end                       
                temp = dist (i,j)/dist(i,k);
                denom=denom+ temp^(2/(m-1));
            end
            if done~=1 %point was a center, hence already assigned
                u_mat(i,j)=1.0/denom;
            end
        end
    end 

end

function [features,labels]=read_iris_data(filename, col_feature1, col_feature2, col_labels) 
    %read iris data set
    fprintf("Working on IRIS data set\n");
 
    %read dataset into table
    data=readtable(filename);
    
    %retrieve data into arrays.
    features=data{:,[col_feature1, col_feature2]}; 
    labels= data{:,col_labels};
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