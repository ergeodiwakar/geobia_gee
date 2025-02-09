var roi:     #(Insert yours study area)
var road:    #(Insert yours road)
var trainingdata: #(Insert your inventory data of landslide area)
var validationdata: #(Insert yours Inventory data which is valide in field)
var landcover: #(Insert yours few landcover information such as water, settlement, cropland, forest etc )
var dataset:  #(Insert the dataset you have prepared in earlier file GEOBIA-DATASET)



Map.addLayer(roi,{},'Study Area')
Map.centerObject(roi,10);
Map.addLayer(road,{color:'red'},'China-Nepal Friendship Highway')
Map.centerObject(roi,12);
Map.addLayer(dataset, {min: 0.0,max: 0.3,bands: ['B4', 'B3', 'B2'],}, 'RGB');

//Palette for the classification
var palette = [ 
  '34A742', //(0)  Forest (Non-landslide)
  '9AF506', //(1)  Cropland (Non-landslide)
  'D8F506', //(2)  Barrenland (Non-landslide)
  'F903E2', //(3)  Settlement (Non-landslide  
  '3399FF', //(4)  Water (Non-landslide)    
  'F50623', ///5)  Landslide
];  

// .................................................................................................................................................................................
//Training data
//Creation of the "training data" feature collection using the pixels having a feature property called "Symbol ID" 
//To improve the information using a buffer with a fixed radius  ( radius = 10 m)
var buffer = function(feature) {
return feature.buffer(10)};
trainingdata = trainingdata.map(buffer)

// If you decided to use the SVM algorithm it's mandatory the normalization of the input bands
var image = ee.Image(dataset);

// calculate the min and max value of an image
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 10,
  maxPixels: 10e13,
}); 

// use unit scale to normalize the pixel values
var dataset = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());

//***************************************************************************************************************************************************************************************************************************************************************************************
//GEOGRAPHICAL OBJECT BASED IMAGE ANALYSIS (GEOBIA)
//***************************************************************************************************************************************************************************************************************************************************************************************
//A) Object Based Image Segmentation (SNIC Segmentation)
//***************************************************************************************************************************************************************************************************************************************************************************************

//Define the superpixel seed location spacing, in pixels
var size_segmentation = 5 // (Vary the size of segmentation as 5,10,15,20)
var size_compactness = 0.0 // (Vary the size of compactness as 0.0, 0.1, 0.2)

// Segmentation using a SNIC approach based on the dataset previosly generated
var seeds = ee.Algorithms.Image.Segmentation.seedGrid(size_segmentation);

var snic = ee.Algorithms.Image.Segmentation.SNIC({
  image: dataset, 
  size:size_segmentation,
  compactness: size_compactness,  
  connectivity: 8, 
  neighborhoodSize: 256, 
  seeds: seeds
})

//***************************************************************************************************************************************************************************************************************************************************************************************
//B) GLCM Feature Extraction 
//***************************************************************************************************************************************************************************************************************************************************************************************

//Create and rescale a grayscale image for GLCM according to Tassi & Vizzari, 2020 
var gray = dataset.expression(
      '(0.3 * NIR) + (0.59 * R) + (0.11 * G)', {
      'NIR': dataset.select('B8'),
      'R': dataset.select('B4'),
      'G': dataset.select('B3')
}).rename('gray');

// the glcmTexture size (in pixel) can be adjusted considering the spatial resolution and the object textural characteristics
var glcm = gray.unitScale(0, 0.50).multiply(100).toInt().glcmTexture({size: 2});

//Define the GLCM indices used in input for the PCA
var glcm_bands= ["gray_asm","gray_contrast","gray_corr","gray_ent","gray_var","gray_idm","gray_savg"]

//Before the PCA the glcm bands are scaled
var image = glcm.select(glcm_bands);

// calculate the min and max value of an image
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 20,
  maxPixels: 10e13,
}); 
var glcm = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());

//**************************************************************************************************************************************
//CALCULATION OF GLCM FEATURE IMPORTANCE
//**************************************************************************************************************************************
var training = image.sampleRegions({
  collection: landcover, 
  properties: ['SymbolID'], 
  scale: 10
});

// Train a classifier (Calculating importance factors for GLCM features using an CART model is not as straightforward as with models like Random Forest.)
var classifier = ee.Classifier.smileRandomForest(50).train({
  features: training,  
  classProperty: 'SymbolID', 
  inputProperties: image.bandNames()
});

// Run .explain() to see what the classifer looks like
print(classifier.explain())

// Calculate variable importance
var importance = ee.Dictionary(classifier.explain().get('importance'))


// Calculate relative importance
var sum = importance.values().reduce(ee.Reducer.sum())

var relativeImportance = importance.map(function(key, val) {
   return (ee.Number(val).multiply(100)).divide(sum)
  })
print(relativeImportance)

// Create a FeatureCollection so we can chart it
var importanceFc = ee.FeatureCollection([
  ee.Feature(null, relativeImportance)
])

var chart = ui.Chart.feature.byProperty({
  features: importanceFc
}).setOptions({
      title: 'Feature Importance',
      vAxis: {title: 'Importance'},
      hAxis: {title: 'Feature'},
      legend: {position: 'none'}
  })
print(chart)

//***************************************************************************************************************************************************************************************************************************************************************************************************
//C) PCA Dimension Reduction 
//***************************************************************************************************************************************************************************************************************************************************************************************************

// Get some information about the input to be used later.
var scale = glcm.projection().nominalScale();
var bandNames = glcm.bandNames();

// Mean center the data to enable a faster covariance reducer and an SD stretch of the principal components.
var meanDict = glcm.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi, 
    scale: scale,
    maxPixels: 10e13
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = glcm.subtract(means);

// This helper function returns a list of new band names.
var getNewBandNames = function(prefix) {
  var seq = ee.List.sequence(1, bandNames.length());
  return seq.map(function(b) {
    return ee.String(prefix).cat(ee.Number(b).int());
  });
};

// This function accepts mean centered imagery, a scale and a region in which to perform the analysis. 
// It returns the Principal Components (PC) in the region as a new image.
var getPrincipalComponents = function(centered, scale, region) {
  // Collapse the bands of the image into a 1D array per pixel.
  var arrays = centered.toArray();
  
  // Compute the covariance of the bands within the region.
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region,
    scale: scale, 
  });
  
  // Get the 'array' covariance result and cast to an array.
  // This represents the band-to-band covariance within the region.
  var covarArray = ee.Array(covar.get('array'));
  
  // Perform an eigen analysis and slice apart the values and vectors.
  var eigens = covarArray.eigen();
  
  // This is a P-length vector of Eigenvalues.
  var eigenValues = eigens.slice(1, 0, 1);
  
  // This is a PxP matrix with eigenvectors in rows.
  var eigenVectors = eigens.slice(1, 1);
    
  // Convert the array image to 2D arrays for matrix computations.
  var arrayImage = arrays.toArray(1);
    
  // Left multiply the image array by the matrix of eigenvectors.
  var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
    
  // Turn the square roots of the Eigenvalues into a P-band image.
  var sdImage = ee.Image(eigenValues.sqrt())
    .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
  
  // Turn the PCs into a P-band image, normalized by SD.
  return principalComponents
    // Throw out an an unneeded dimension, [[]] -> [].
    .arrayProject([0])
    // Make the one band array image a multi-band image, [] -> image.
    .arrayFlatten([getNewBandNames('pc')])
    // Normalize the PCs by their SDs.
    .divide(sdImage);
};

// Get the PCs at the specified scale and in the specified region
var pcImage = getPrincipalComponents(centered, scale, roi);

//Select the band "clusters" from the snic output fixed on its scale of 5 meters and add them the PC1 taken from the PCA data.
// Calculate the mean for each segment with respect to the pixels in that cluster
var clusters_snic = snic.select("clusters")

//**************************************************************************************************************************************************************************************************
//VISUALIZING SNIC SEGMENTATION
//**************************************************************************************************************************************************************************************************
var vectors = clusters_snic.reduceToVectors({
  geometryType: 'polygon',
  reducer: ee.Reducer.countEvery(),
  scale: 20,
  maxPixels: 1e13,
  geometry: roi,
});

var empty = ee.Image().byte();

var outline = empty.paint({
  featureCollection: vectors,
  color: 0.25,
  width: 0.25,
});
Map.addLayer(outline, {palette: 'FF0000'}, 'segments');

//***************************************************************************************************************************************************************************************************
//VISUALIZING GLCM FEATURE EXTRACTION
//***************************************************************************************************************************************************************************************************
clusters_snic = clusters_snic.reproject ({crs: clusters_snic.projection (), scale: 30});
// Map.addLayer(clusters_snic.randomVisualizer(), {}, 'clusters')

//*********************************************************************************************************************************************************************************************************************************************************************************************************
//MACHINE LEARNING MODEL PERFORMANCE
//*********************************************************************************************************************************************************************************************************************************************************************************************************
var new_feature = clusters_snic.addBands(pcImage.select("pc1"))

var new_feature_mean = new_feature.reduceConnectedComponents({
  reducer: ee.Reducer.mean(),
  labelBand: 'clusters'
})

//Create a dataset with the new band used so far together with the band "clusters" and their new mean parameters
var final_bands = new_feature_mean.addBands(snic) 

//Define the training bands removing just the "clusters" bands
var predictionBands=final_bands.bandNames().remove("clusters")

//if you want use RandomForest (classifier_alg= "RF") or use SVM (classifier_alg= "SVM") else use CART (classifier_alg= "CART")
var classifier_alg="RF"

var minMax = final_bands.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 100,
  maxPixels: 10e9,
});
var final_bands = ee.ImageCollection.fromImages(
  final_bands.bandNames().map(function(name){
    name = ee.String(name);
    var band = final_bands.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(final_bands.bandNames());


//Training bands with predictionBands
var training_geobia = final_bands.select(predictionBands).sampleRegions({
  collection: trainingdata,
  properties: ['SymbolID'],
  scale: 50
});

// Define the bands to be used along with PCA (GLCM indices)
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'].concat(glcm_bands);

// Merge the PC image and the GLCM indices
var inputImage = pcImage.addBands(glcm);

// Get the feature properties for the trainingdata feature collection
var training = inputImage.sampleRegions({
  collection: trainingdata, 
  properties: ['SymbolID'], 
  scale: 30
});

//Training the classifier
if(classifier_alg=="RF"){
  var RF = ee.Classifier.smileRandomForest(100).train({
  features:training_geobia, 
  classProperty:'SymbolID', 
  inputProperties: predictionBands
});
}
else if (classifier_alg == "SVM") {
    RF = ee.Classifier.libsvm({
        kernelType: 'RBF',
        gamma: 1,
        cost: 10
    }).train({
        features: training_geobia,
        classProperty: 'SymbolID',
        inputProperties: predictionBands
    });
}
else if (classifier_alg=="CART") {
 var RF = ee.Classifier.smileCart().train({
    features: training_geobia, 
    classProperty: 'SymbolID', 
    inputProperties: predictionBands
  });
}
else{
  print("You need to set your classifier for the Object based approach")
}

var classy_RF = final_bands.select(predictionBands).classify(RF);
classy_RF = classy_RF.reproject ({crs: classy_RF.projection (), scale: 50});

//************************************************************************************************************************************************************************************************************************************************************************************
//VISUALIZATION OF LANDSLIDE LOCATION
//************************************************************************************************************************************************************************************************************************************************************************************
// Create a connected components image using the labels obtained from GLCM clustering
var segmentLines = final_bands.select("clusters");

// Get the classified result (landslide areas) from the machine learning model
var classifiedResult = classy_RF.clip(roi).select('classification');

// Create a mask for the landslide area
var landslideMask = classifiedResult.eq(5); // Assuming class 5 represents the landslide in the classification

// Apply the mask to the segment lines image
var landslideSegmentLines = segmentLines.updateMask(landslideMask);

// Visualize the segment lines in the landslide area (optional)
var segmentLinesVis = {palette: 'ff0000'};
Map.addLayer(landslideSegmentLines, segmentLinesVis, 'Landslide Area');

//******************************************************************************************************************************************************************************************************************************************************************
//VALIDATION OF THE MODEL
//******************************************************************************************************************************************************************************************************************************************************************
//Validation of the object-oriented approach
var classifier_geobia = final_bands.select(predictionBands).sampleRegions({
  collection: validationdata,
  properties: ['SymbolID'],
  scale: 30
});

var classificazione = classifier_geobia.classify(RF);
var testAccuracy = classificazione.errorMatrix('SymbolID', 'classification');

// True positive (TP)
var TP = ee.Array(testAccuracy.array()).get([1, 1]);

// False positive (FP)
var FP = ee.Array(testAccuracy.array()).get([0, 1]);

// True negative (TN)
var TN = ee.Array(testAccuracy.array()).get([0, 0]);

// False negative (FN)
var FN = ee.Array(testAccuracy.array()).get([1, 0]);

// Accuracy
var accuracy = TP.add(TN).divide(TP.add(TN).add(FP).add(FN)).multiply(100);

// Precision
var precision = TP.divide(TP.add(FP)).multiply(100);

// Recall
var recall = TP.divide(TP.add(FN)).multiply(100);

// Specificity
var specificity = TN.divide(TN.add(FP)).multiply(100);

//F1 Score
var f1Score = precision.multiply(recall).multiply(2).divide(precision.add(recall));

//***************************************************************************************************************************************************************************************************************************************
// PRINT THE MODEL EVALUATION RESULTS
//***************************************************************************************************************************************************************************************************************************************
print('GEOBIA approach_Test confusion matrix: ', testAccuracy);  
print('GEOBIA APPROACH:Overall Accuracy ', testAccuracy.accuracy().multiply(100));
print('GEOBIA APPROACH:Kappa ', testAccuracy.kappa().multiply(100));
print('GEOBIA APPROACH:Precision', precision);
print('GEOBIA APPROACH:Recall', recall);
print('GEOBIA APPROACH:Specificity', specificity);
print('GEOBIA APPROACH:F1 Score', f1Score);

//**********************************************************************************************************************************************************************************************************************************************
//PRINT THE NUMBER OF PIXELS FOR EACH CLASS
//**********************************************************************************************************************************************************************************************************************************************
var analysis_image_sl = classy_RF.select("classification")
var class0 =  analysis_image_sl.updateMask(analysis_image_sl.eq(0))
var class1 =  analysis_image_sl.updateMask(analysis_image_sl.eq(1))
var class2 =  analysis_image_sl.updateMask(analysis_image_sl.eq(2))
var class3 =  analysis_image_sl.updateMask(analysis_image_sl.eq(3))
var class4 =  analysis_image_sl.updateMask(analysis_image_sl.eq(4))
var class5 =  analysis_image_sl.updateMask(analysis_image_sl.eq(5))

var all = class0.addBands(class1).addBands(class2).addBands(class3).addBands(class4).addBands(class5)

var count_pixels = all.reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: roi,
  scale:6500,
  maxPixels: 1e13,
  })

print(count_pixels, "GEOBIA APPROACH: pixels for each class")

//********************************************************************************************************************************************************************************************************************************************
//COUNT THE NO.OF PIXELS FOR LANDSLIDE
//********************************************************************************************************************************************************************************************************************************************
// Reduce the classified image to a single value representing the count of landslide pixels
var landslideCount = classy_RF.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: roi,
  scale: 100,
  maxPixels: 1e9,
  tileScale: 16,
});

// Get the count value
var landslidePixelCount = ee.Number(landslideCount.get('classification')).int();

//******************************************************************************************************************************************************************************************************************************************
//CALCULATION OF LADNSLIDE LOCATIONS
//******************************************************************************************************************************************************************************************************************************************
var landslideColor = palette.indexOf('F50623'); // Color code for landslide class in the palette

// Get the classified image
var classifiedImage = classy_RF;

// Create a binary mask for landslide pixels within the ROI
var landslideMask = classifiedImage.eq(landslideColor);

// Count the number of landslide pixels within the ROI
var landslidelocationCount = landslideMask.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: roi,
  scale: 100,
  bestEffort: true,
  tileScale: 16,
}).get('classification');

print('Number of landslide locations:', landslidelocationCount);

//*********************************************************************************************************************************************************************************************************************************************************************************
//DATA EXPORT
//**********************************************************************************************************************************************************************************************************************************************************************************
// Export the boundary line image to Google Drive
Export.image.toDrive({
  image: boundaryImage,
  description: 'Landslide_Boundary',
  folder: 'APRIL_2024',
  scale: 20, // Adjust the scale according to your preference
  region: roi, // Specify the region of interest
  fileFormat: 'GeoTIFF', // Choose the file format you prefer
});

//..............................................................................................................................................................................................................
// Export to Drive
Export.image.toDrive({
  image:landslideSegmentLines,
  description: 'GOBIA_RF_CLASSIFIED',
  region : roi,
  folder: 'CLASSIFIED OUTPUT',
  fileFormat: 'GEOTIFF',
  scale: 20,
  maxPixels:1e13,
});