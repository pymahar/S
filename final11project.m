clear;
clc;

datasetPath = 'Copy_of_project';  
imgSize = [128 128];             


fprintf('Loading dataset and training classifier...\n');

classes = dir(datasetPath);
classes = classes([classes.isdir] & ~startsWith({classes.name}, '.'));

features = [];
labels = [];

for i = 1:length(classes)
    className = classes(i).name;
    classPath = fullfile(datasetPath, className);

    imgFiles = [dir(fullfile(classPath, '*.jpeg')); dir(fullfile(classPath, '*.png'))];

    for j = 1:length(imgFiles)
        img = imread(fullfile(classPath, imgFiles(j).name));
        if size(img, 3) == 3
            img = rgb2gray(img);  
        end
        img = imresize(img, imgSize);
        img = histeq(img);       

        hogFeatures = extractHOGFeatures(img);

        features = [features; hogFeatures];
        labels = [labels; string(className)];
    end
end

labels = categorical(labels);  % Convert labels to categorical for classification
SVMModel = fitcecoc(features, labels);  % Train SVM classifier

fprintf('Model trained successfully.\n');

choice = menu('Choose input method:',  'Load image from disk');

switch choice
    
    case 1
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png'}, 'Select a signature image');
        if isequal(file, 0)
            disp(' No image selected.');
            return;
        end
        img = imread(fullfile(path, file));
    otherwise
        disp(' No option selected.');
        return;
end

if size(img, 3) == 3
    img = rgb2gray(img);
end
img = imresize(img, imgSize);
img = histeq(img);  % Enhance contrast

hogFeatures = extractHOGFeatures(img);
predictedLabel = predict(SVMModel, hogFeatures);

figure;
imshow(img);
title([' Predicted owner: ', char(predictedLabel)], 'FontSize', 14, 'Color', 'b');
fprintf(' This signature likely belongs to: %s\n', predictedLabel);
