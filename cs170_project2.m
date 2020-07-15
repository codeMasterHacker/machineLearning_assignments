%Assume the following:
% Will only be given datasets that have two classes
% Will only be given datasets that have continuous features
% There may be an arbitrary number of features (for simplicity I will cap the maximum number at 64)
% There may be an arbitrary number of instances (rows), for simplicity I will cap the maximum number at 2,048
% First column is the class and these values will always be either "1"s or "2"s
% The other columns contain the features, which are not normalized

fprintf("\nWelcome to Enrique Munoz's Feature Selection Algorithm.\n");
fileName = string(input("Type in the name of the file to test: ", "s"));

try
    data = importdata(fileName," ");
catch
    fprintf("\nThis file doesn't exist. Exiting...\n");
    return
end

fprintf("\nThis dataset has %i features (not including the class attribute), " + ...
        "with %i instances.\n", (size(data,2)-1), size(data,1));

fprintf("\nPlease wait while I normalize the data...");
classColumn = data(:,1);
featureColumns = data(:,2:end);
normalizedData = zscore(featureColumns, [], 1); %zscore normalization along the columns
data = [classColumn, normalizedData];
fprintf("Done!\n");

algorithm = input("\nType the number of the algorithm you want to run.\n" + ...
                  "1) Forward Selection\n" + ...
                  "2) Backward Elimination\n");

switch (algorithm)
    case 1
        currentAccuracy = defaultRate(classColumn, size(data,1));
        
        fprintf("\nRunning default rate, i.e. no features selected, I get an accuracy of %.2f%%\n\n" + ...
                "Beginning Forward Selection Search\n\n", currentAccuracy*100);
            
        [featureSubset, maxAccuracy] = forwardSelection_search(data, currentAccuracy);
        featureSubset = sort(featureSubset(featureSubset > 0));
    case 2
        currentAccuracy = leave_one_out_validator(uint8([1:(size(data,2)-1)]), featureColumns, classColumn);
        
        fprintf("\nRunning nearest neighbor with all %i features, using “leaving-one-out” evaluation, I get an accuracy of %.2f%%\n\n" + ...
            "Beginning Backward Elimination Search\n\n", (size(data,2)-1), currentAccuracy*100);
        
        [featureSubset, maxAccuracy] = backwardElimination(data, currentAccuracy);
    otherwise
        fprintf("Entered an invlaid option. Exiting...\n");
        return;
end

format=['Finished search! The best feature subset is {' repmat('%i,',1,numel(featureSubset))];
fprintf(format,featureSubset);
fprintf("}, which has an accuracy of %.2f%%\n", maxAccuracy*100);

numFeatures = numel(featureSubset);
if (numFeatures > 3)
    fprintf("\nThe best feature subset has 4 or more dimension, ergo cannot plot this.\n");
else
    featureSubset = featureSubset + 1;
    featureSubset = [1, featureSubset];
    dataPoints = data(:,featureSubset);
        
    class1_dataPoints = dataPoints(dataPoints(:,1) == 1, 2:end);
    class2_dataPoints = dataPoints(dataPoints(:,1) == 2, 2:end);
    
    switch (numFeatures)
    case 1 
        plot(class1_dataPoints(:,1)', [1:numel(class1_dataPoints)], "ro", "MarkerFaceColor", "r");
        hold on;
        plot(class2_dataPoints(:,1)', [1:numel(class2_dataPoints)], "ko", "MarkerFaceColor", "k");
        hold off;
    case 2
        plot(class1_dataPoints(:,1)', class1_dataPoints(:,2)', "ro", "MarkerFaceColor", "r");
        hold on;
        plot(class2_dataPoints(:,1)', class2_dataPoints(:,2)', "ko", "MarkerFaceColor", "k");
        hold off;
        ylabel(["Feature ", num2str(featureSubset(3)-1)]);
    case 3
        plot3(class1_dataPoints(:,1)', class1_dataPoints(:,2)', class1_dataPoints(:,3)', "ro", "MarkerFaceColor", "r");
        hold on;
        plot3(class2_dataPoints(:,1)', class2_dataPoints(:,2)', class2_dataPoints(:,3)', "ko", "MarkerFaceColor", "k");
        hold off;
        box on;
    end
    
    title("Data Points");
    xlabel(["Feature ", num2str(featureSubset(2)-1)]);
    legend("clas 1", "class 2");
end



function accuracy = defaultRate(classColumn, dataSize)
%Since no feature have been selected, compute the current accuracy using the default rate.
%Default Rate = size(most common class) / size(dataset)

mostCommonClass = mode(classColumn);
mostCommon_classArray = classColumn(classColumn == mostCommonClass);
accuracy = size(mostCommon_classArray,1) / dataSize;
end



function [featureSubset, maxAccuracy] = backwardElimination(data, maximumAccuracy)
classColumn = data(:,1);
featureColumns = data(:,2:end);

numFeatures = size(data,2)-1;
numAccuracies = numFeatures;

featuresSet = [1:numFeatures];

previousAccuracy = maximumAccuracy;

for i = 1:numFeatures
    accuracies = zeros(1, numAccuracies, "double");
    
    for j = 1:numAccuracies
        fSet = featuresSet;
        fSet(j) = [];
        
        if (isempty(fSet))
            accuracies(j) = defaultRate(classColumn, size(data,1));
        else
            accuracies(j) = leave_one_out_validator(fSet, featureColumns, classColumn);
        end
        
        fmt=['\tUsing feature(s) {' repmat('%i,',1,numel(fSet))];
        fprintf(fmt,fSet);
        fprintf("}, with an accuracy of %.2f%%\n", accuracies(j)*100);
    end
    
    [currentAccuracy, index] = max(accuracies);
    featuresSet(index) = [];
    
    if (currentAccuracy >= maximumAccuracy)
        maximumAccuracy = currentAccuracy;
        maxAccuracy = maximumAccuracy;
        featureSubset = featuresSet;
    elseif (currentAccuracy < previousAccuracy)
        fprintf("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)");
    end
    
    numAccuracies = numAccuracies - 1;
    
    previousAccuracy = currentAccuracy;
    
    fmt=['\nFeature set {' repmat('%i,',1,numel(featuresSet))];
    fprintf(fmt,featuresSet);
    fprintf("} was best, with an accuracy of %.2f%%\n\n", currentAccuracy*100);
end

end



function [featureSubset, maxAccuracy] = forwardSelection_search(data, maximumAccuracy)
classColumn = data(:,1);
featureColumns = data(:,2:end);
 
numFeatures = size(data,2)-1;
 
fSubset = zeros(1, numFeatures, "uint8");
accuracies = zeros(1, numFeatures, "double");
features = zeros(1, numFeatures, "uint8");
 
previousAccuracy = maximumAccuracy;
 
for i = 1:numFeatures
    featuresSet = fSubset(1:i); k=0;
     
    for j = 1:numFeatures
        if (~ismember(j, fSubset))
            featuresSet(i) = j;
            fSet = sort(featuresSet); k=k+1;
             
            accuracies(k) = leave_one_out_validator(fSet, featureColumns, classColumn);
            features(k) = j;
             
            fmt=['\tUsing feature(s) {' repmat('%1.0f,',1,numel(fSet))];
            fprintf(fmt,fSet);
            fprintf("}, with an accuracy of %.2f%%\n", accuracies(k)*100);
        end
    end
     
    [currentAccuracy, index] = max(accuracies);
    fSubset(i) = features(index);
     
    accuracies(:) = 0;
    features(:) = 0;
     
    if (currentAccuracy >= maximumAccuracy)
        maximumAccuracy = currentAccuracy;
        maxAccuracy = maximumAccuracy;
        featureSubset = fSubset;
    elseif (currentAccuracy < previousAccuracy)
        fprintf("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)");
    end
     
    previousAccuracy = currentAccuracy;
     
    set = fSubset(fSubset > 0);
    fmt=['\nFeature set {' repmat('%1.0f,',1,numel(set))];
    fprintf(fmt,set);
    fprintf("} was best, with an accuracy of %.2f%%\n\n", currentAccuracy*100);    
end
 
end



function accuracy = leave_one_out_validator(featuresSet, features, classColumn)
featureColumns = features(:,featuresSet);
accuracies = zeros(1, size(classColumn,1), "double");

for i = 1:size(accuracies,2)
    testPoint = featureColumns(i, :);
    testPointClass = classColumn(i, :);
    
    trainingPoints = featureColumns;
    trainingPoints(i, :) = [];
    
    trainingPoints_classes = classColumn;
    trainingPoints_classes(i, :) = [];
    
    diffMatrix = trainingPoints - repmat(testPoint, size(trainingPoints, 1), 1);
    distances = sqrt(sum(diffMatrix.^2,2));
    
    [~, index] = min(distances);
    class = trainingPoints_classes(index);
    
    if class == testPointClass
        accuracies(i) = 1;
    else
        accuracies(i) = 0;
    end
    
end

accuracy = sum(accuracies) / size(accuracies, 2);

end