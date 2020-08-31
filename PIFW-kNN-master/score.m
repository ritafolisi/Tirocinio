function acc = score(testMatrix, trainMatrix)

    matched = 0; unmatched = 0;
    
    % Prediction and checking of labels
    for i = 1:size(testMatrix,1)
		lab = predict(trainMatrix,testMatrix(i,:));
        if lab == testMatrix(i,size(testMatrix,2))
			matched = matched + 1;
		else
			unmatched = unmatched+1;
        end
    end
    
    h = sprintf('Matched: %d', matched);
    disp(h)
    
    acc = matched/(matched+unmatched);
    acc = acc*100;
    
end