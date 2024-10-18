%% Negative class

neg = dir('./CaltechFaces/my_train_non_face_scenes/*.jpg');

%% Negative Class Augmentation (Extended)


mkdir('./CaltechFaces/my2_train_non_face_scenes/')
outdir = './CaltechFaces/my2_train_non_face_scenes';


%{

for ii = 1:size(neg,1)
    % original images
    im = imread([neg(ii).folder filesep neg(ii,1).name]);
    imwrite(im,[outdir filesep neg(ii,1).name]); 
    
    [pp,ff,ee] = fileparts(neg(ii).name); % Extract file parts

    % Vertical flipping
    im_flip = fliplr(im); 
    imwrite(im_flip,[outdir filesep ff '_flip' ee]); 
    
    % Horizontal flipping 
    im_flip = flipud(im); 
    imwrite(im_flip,[outdir filesep ff '_flipud' ee]); 

    % Rotation (10 random rotations)
    for nrot = 1:10
        random_angle = 360*rand(1); % Generate a random angle between 0 and 360 degrees
        imr = imrotate(im, random_angle, 'crop'); 
        imwrite(imr, [outdir filesep ff '_r' num2str(nrot) ee]);
    end
    
    % Random cropping (6 random croppings)
    for ncrop = 1:6
        cropSize = [randi([round(size(im,1)*0.5), size(im,1)]), randi([round(size(im,2)*0.5), size(im,2)])]; % Random crop size
        rand_y = randi([1, size(im,1)-cropSize(1)+1]); % Randomly select crop position y
        rand_x = randi([1, size(im,2)-cropSize(2)+1]); % Randomly select crop position x
        im_crop = im(rand_y:rand_y+cropSize(1)-1, rand_x:rand_x+cropSize(2)-1, :); % Perform cropping
        imwrite(im_crop, [outdir filesep ff '_c' num2str(ncrop) ee]); % Save cropped image
    end
    
    % Random scaling (6 random zoom in/out)
    for nscale = 1:6
        scaleFactor = 1 + rand(1)*0.5 - 0.25; % Random scale factor between 0.75 and 1.25
        im_scaled = imresize(im, scaleFactor); % Perform scaling
        imwrite(im_scaled, [outdir filesep ff '_s' num2str(nscale) ee]); % Save scaled image
    end
    
    % Brightness adjustment (6 random adjustments)
    for nbright = 1:6
        brightnessOffset = randi([-50, 50]); % Random brightness offset between -50 and 50
        im_bright = im + brightnessOffset; % Perform brightness adjustment
        imwrite(im_bright, [outdir filesep ff '_b' num2str(nbright) ee]); % Save brightness adjusted image
    end
    
    % Gaussian noise addition (6 random g additions)
    for nnoise = 1:6
        noiseLevel = randi([5, 20]); % Random noise level between 5 and 20
        noise = randn(size(im)) * noiseLevel; % Generate Gaussian noise
        im_noisy = im + uint8(noise); % Add noise to image
        imwrite(im_noisy, [outdir filesep ff '_n' num2str(nnoise) ee]); % Save noisy image
    end
end


%}

%%

negativeFolder = './CaltechFaces/my2_train_non_face_scenes';
negativeImages = imageDatastore(negativeFolder);

%% positive class
faces = dir('./CaltechFaces/my_train_faces/*.jpg');
sz = [size(faces,1) 2];
varTypes = {'cell','cell'};
varNames = {'imageFilename','face'};
facesIMDB = table('Size',sz,'VariableTypes',varTypes,'VariableNames', varNames);

for ii=1:size(faces,1)
    facesIMDB.imageFilename(ii) = {[faces(ii).folder filesep faces(ii).name]};
    facesIMDB.face(ii) = {[1 1 32 32]};
end 

positiveInstances = facesIMDB;

%% V3 detector training

tic
trainCascadeObjectDetector("myFaceDetector.xml",positiveInstances,...
    negativeFolder,ObjectTrainingSize="auto", NegativeSamplesFactor = 2,...
    NumCascadeStages=10, FalseAlarmRate=0.01, TruePositiveRate=0.999, ...         
    FeatureType='LBP'); %HOG,LBP,Haar
toc

%% Visualize the results 

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
%detector = vision.CascadeObjectDetector(); matlab detector

imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector, img); %prediction: detected or not

    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg = imresize(detectedImg, 800/max(size(detectedImg)));

    figure(1), clf
    imshow(detectedImg)
    %waitforbuttonpress
end 

close all

%% Compute our results

load('./CaltechFaces/test_scenes/GT.mat');

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
%detector = vision.CascadeObjectDetector(); matlab detector

imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');


numImages = size(imgs, 1);
results = table('Size',[numImages 2],...
    'VariableTypes', {'cell','cell'},...
    'VariableNames',{'face','Scores'});

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector, img);
    results.face{ii}=bbox;
    results.Scores{ii}=0.5+zeros(size(bbox,1),1);
end

%waitforbuttonpress

%% Compute average precision

[ap, recall, precision] = evaluateDetectionPrecision(results, GT,0.2);
figure(2),clf
plot(recall, precision, 'r', LineWidth=2)
xlim([0 1])
ylim([0 1])
grid on
title(sprintf('Average Precision = %.2f',ap)) 
waitforbuttonpress


%% Show both detected faces and real test faces

load('./CaltechFaces/test_scenes/GT.mat');
imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');
detector = vision.CascadeObjectDetector('myFaceDetector.xml');

for i=1:size(GT,1)

    img = imread([imgs(i).folder filesep imgs(i).name]);
    n_faces = numel(GT{i,:}{1})/4;                   % n of faces in img
    bbox = step(detector, img);                      % box detected

    for ii=1:n_faces
        cordinates = GT{i,:}{1}(ii,:);
        x = cordinates(1);
        y = cordinates(2);
        z = cordinates(3);
        w = cordinates(4);

        img = insertShape(img, 'Rectangle', [x y z w], 'LineWidth', 1,...
              'Color', 'red'); % real boxes
        
        %{
        cntr_x = x + z/2;
        cntr_y = y + w/2;                 % circle instead of rectangle
        rad = max(z, w)/1.2; 
        img = insertShape(img, 'Circle', [cntr_x, cntr_y, rad], 'LineWidth', 1, 'Color', 'red');
        %}

    end  

    if ~isempty(bbox)
        for j = 1:size(bbox, 1)
            img = insertObjectAnnotation(img,'rectangle',bbox,'face'); % detected boxes
        end
    end

    img = imresize(img, 800/max(size(img)));
    imshow(img);
    waitforbuttonpress
end

