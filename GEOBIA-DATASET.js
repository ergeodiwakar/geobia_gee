var roi: Table users/khadkadiwakar/roi #(Insert yours research area)
var road: Table users/khadkadiwakar/road #(Insert yours road)
var landcover: Table projects/ee-khadkadiwakar-chapter1/assets/Landcover_Data_set #(Insert yours Landcover data)



//Research Area Defination
Map.addLayer(roi,{color:'blue'},'Study area');
Map.centerObject(roi,9);

//********************************************************************************************************************
//1.Define Vegetation InBands Indices
//********************************************************************************************************************
var period_of_interest = ee.Filter.date('2016-01-01', '2022-12-31');
var inBands = ["B2","B3","B4","B6","B8","B11"]
var outBands = inBands.concat("NDVI","NDWI","SAVI","NDMI","EVI")
Map.centerObject(roi,12);

//Function to mask the clouds in Sentinel-2
function maskS2clouds(image) {
  var qa = image.select('QA60');

// Bits 8 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 8;
  var cirrusBitMask = 1 << 11;

// Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

//Load Sentinel-2A Real Time Series imagery between 2016 to 2022, filtered by date, bounds and percentage of cloud cover 
var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')//Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(maskS2clouds);
                  
print("Sentinel 2 Image Collection",dataset)

// Create a median image from the ImageCollection
var medianImage = dataset.median();

// Display the median image
Map.addLayer(medianImage.clip(roi), { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3 }, 'Median Image');

print("Sentinel 2 Median Image Collection",medianImage)

//*************************************************************************************************************************
//2. Defination and Calculation of Topographical Indices
//*************************************************************************************************************************
//Load NASA NASADEM Digital Elevation 30m Image set
var dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation');

// 2.1) Calculation of slope. Units are degrees, range is [0,90).
var slope = ee.Terrain.slope(dem).clip(roi);

// Set Slope visualization properties
Map.addLayer(slope, {min: 0, max: 89.99, palette: ['red', 'yellow', 'blue', 'green']}, 'Slope');

//2.2) Calculation of Elevation 
var elevation = dem.clip(roi);

// Set elevation visualization properties
var elevationVis = {min: 0, max: 6500,palette: ['green', 'blue', 'yellow','purple', 'red']};

// Set elevation <= 0 as transparent and add to the map.
Map.addLayer(elevation.updateMask(elevation.gt(0)), elevationVis, 'Elevation');

//2.3) Calculation of the Aspect
var aspect = ee.Terrain.aspect(dem).clip(roi);

// Display Aspect layers on the map.
Map.addLayer(aspect, {min: 0, max: 359.99, palette: ['red', 'yellow', 'blue', 'green']}, 'Aspect');

//2.4) Hillshade is calculated based on illumination azimuth=270, elevation=45
var hillshade = ee.Terrain.hillshade(dem, 270, 45).clip(roi);

// Display hillshade layers on the map
Map.addLayer(hillshade.select('hillshade'), {min: 0, max: 255,palette: ['red', 'yellow', 'blue', 'green']}, 'Hillshade');

//******************************************************************************************************************************
//3.Add Vegetational Spectral Indices Such as NDVI,NDWI,SAVI,NDMI and EVI
//******************************************************************************************************************************

// 3.1)Add NDVI Spectral Indices....................................................
//Add NDVI Spectral Indices
var addNDVI = function(image) {var ndvi = image.normalizedDifference(['B8', 'B4'])
  .rename('NDVI')
  .copyProperties(image,['system:time_start']);
  return image.addBands(ndvi);
};

// Load the Sentinel-2 image collection, filter by date and roi, and apply the ARVI function
var dataset4 = ee.ImageCollection("COPERNICUS/S2")
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(addNDVI)
              
// Clip the image collection to the ROI
var s5_clipped = dataset4.map(function(image) {
  return image.clip(roi);
});

// Select the NDVI band from the clipped image collection
var ndvi_clipped = s5_clipped.select('NDVI').median();

// Add the clipped NDVI layer to the map
Map.addLayer(ndvi_clipped, {min:0, max:1, palette:['red', 'yellow','blue','green']}, 'NDVI');

// Center the map on the clipped image
Map.centerObject(roi, 10);

//3.2)Add NDWI Spectral Indices........................................
//Add NDWI Spectral Indices
var addNDWI = function(image) {var ndwi = image.normalizedDifference(['B8', 'B3'])
  .rename('NDWI')
  .copyProperties(image,['system:time_start']);
  return image.addBands(ndwi);
};

// Load the Sentinel-2 image collection, filter by date and roi, and apply the ARVI function
var dataset3 = ee.ImageCollection("COPERNICUS/S2")
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(addNDWI)
              
// Clip the image collection to the ROI
var s4_clipped = dataset3.map(function(image) {
  return image.clip(roi);
});

// Select the NDWI band from the clipped image collection
var ndwi_clipped = s4_clipped.select('NDWI').median();

// Add the clipped NDWI layer to the map
Map.addLayer(ndwi_clipped, {min:-0.8, max:0.8, palette:['blue', 'white', 'green']}, 'NDWI');

// Center the map on the clipped image
Map.centerObject(roi, 10);

//3.3)Add SAVI Spectral Indices........................................
var addSAVI = function(image) {var savi = image.expression(
  '(NIR - RED) / (NIR + RED + L) * (1.0 + L)', 
  {
    'RED': image.select('B4'), 
    'BLUE': image.select('B2'),
    'NIR': image.select('B8'),
    'L': 0.5
  }
)
  .rename('SAVI')
  .copyProperties(image,['system:time_start']);
  return image.addBands(savi);
};

// Load the Sentinel-2 image collection, filter by date and roi, and apply the SAVI function
var dataset1 = ee.ImageCollection("COPERNICUS/S2")
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(addSAVI)
                  
// Clip the image collection to the ROI
var s2_clipped = dataset1.map(function(image) {
  return image.clip(roi);
});

// Select the SAVI band from the clipped image collection
var savi_clipped = s2_clipped.select('SAVI').median();

// Add the clipped SAVI layer to the map
Map.addLayer(savi_clipped, {min:0, max:1, palette:['red','yellow','green']}, 'SAVI');

// 3.4)Add NDMI Spectral Indices........................................
var addNDMI = function(image) {
  var ndmi = image.expression(
    '((NIR-SWIR)/(NIR+SWIR))', 
    {
      'NIR': image.select('B8'), 
      'SWIR': image.select('B11')
    }
)
  .rename('NDMI')
  .copyProperties(image,['system:time_start']);
  return image.addBands(ndmi);
};

// Load the Sentinel-2 image collection, filter by date and roi, and apply the NDMI function
var dataset7 = ee.ImageCollection("COPERNICUS/S2")
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(addNDMI)
              
// Clip the image collection to the ROI
var s8_clipped = dataset7.map(function(image) {
  return image.clip(roi);
});

// Select the NDMI band from the clipped image collection
var ndmi_clipped = s8_clipped.select('NDMI').median();

// Add the clipped NDMI layer to the map
Map.addLayer(ndmi_clipped, {min:-1, max:1, palette:['blue', 'green', 'yellow', 'orange', 'red']}, 'NDMI');

// Center the map on the clipped image
Map.centerObject(roi, 10);

// 3.5)Add EVI Spectral Indices........................................
var addEVI = function(image) {
  var evi = image.expression(
    '(2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)))', 
    {
      'NIR': image.select('B8'), 
      'BLUE': image.select('B2'),
      'RED': image.select('B4')
    }
)
  .rename('EVI')
  .copyProperties(image,['system:time_start']);
  return image.addBands(evi);
};

// Load the Sentinel-2 image collection, filter by date and roi, and apply the EVI function
var dataset6 = ee.ImageCollection("COPERNICUS/S2")
                  .filter(period_of_interest)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                  .map(addEVI)
              
// Clip the image collection to the ROI
var s7_clipped = dataset6.map(function(image) {
  return image.clip(roi);
});

// Select the EVI band from the clipped image collection
var evi_clipped = s7_clipped.select('EVI').median();

// Add the clipped EVI layer to the map
Map.addLayer(evi_clipped, {min:-1, max:1, palette:['blue', 'green', 'yellow', 'orange', 'red']}, 'EVI');

// Center the map on the clipped image
Map.centerObject(roi, 10);

//********************************************************************************************************************************
// Collection with all images also containing the vegetation Indices & Topographical Indices
//********************************************************************************************************************************
var collection = dataset.select(inBands).map(addNDVI).map(addNDWI).map(addSAVI).map(addNDMI).map(addEVI)
  .map(function(image) {
// Add Topographical indices
    image = image.addBands(elevation).addBands(slope).addBands(aspect).addBands(hillshade);
    
// Remove duplicate bands
    image = image.select(['B2', 'B3', 'B4', 'B6', 'B8', 'B11', 'NDVI', 'NDWI', 'SAVI', 'NDMI', 'EVI', 'elevation', 'slope', 'aspect', 'hillshade']);
    return image;
  });
print('Collection with inBands, Vegetation indices & Topographical Indices', collection);

// Define and calculate the median bands and the other index statistics
var band_median = collection.select(inBands).median()
var ndvimax = collection.select('NDVI').reduce(ee.Reducer.max()).rename("NDVI_MAX");
var ndvimean = collection.select('NDVI').reduce(ee.Reducer.mean()).rename("NDVI");
var ndvistd = collection.select('NDVI').reduce(ee.Reducer.stdDev()).float().rename("NDVI_STD");
var ndwimax = collection.select('NDWI').reduce(ee.Reducer.max()).rename("NDWI_MAX");
var ndwimean = collection.select('NDWI').reduce(ee.Reducer.mean()).rename("NDWI");
var ndwistd = collection.select('NDWI').reduce(ee.Reducer.stdDev()).float().rename("NDWI_STD");
var savimax = collection.select('SAVI').reduce(ee.Reducer.max()).rename("SAVI_MAX");
var savimean = collection.select('SAVI').reduce(ee.Reducer.mean()).rename("SAVI");
var savistd = collection.select('SAVI').reduce(ee.Reducer.stdDev()).float().rename("SAVI_STD");
var ndmimax = collection.select('NDMI').reduce(ee.Reducer.max()).rename("NDMI_MAX");
var ndmimean = collection.select('NDMI').reduce(ee.Reducer.mean()).rename("NDMI");
var ndmistd = collection.select('NDMI').reduce(ee.Reducer.stdDev()).float().rename("NDMI_STD");
var evimax = collection.select('EVI').reduce(ee.Reducer.max()).rename("EVI_MAX");
var evimean = collection.select('EVI').reduce(ee.Reducer.mean()).rename("EVI");
var evistd = collection.select('EVI').reduce(ee.Reducer.stdDev()).float().rename("EVI_STD");
var slopemax = collection.select('slope').reduce(ee.Reducer.max()).rename("Slope_Max");
var slopemean = collection.select('slope').reduce(ee.Reducer.mean()).rename("SLOPE");
var slopestd = collection.select('slope').reduce(ee.Reducer.stdDev()).float().rename("Slope_STD");
var elevationmax =collection.select('elevation').reduce(ee.Reducer.max()).rename("Elevation_Max");
var elevationmean = collection.select('elevation').reduce(ee.Reducer.mean()).rename("ELEVATION");
var elevationstd = collection.select('elevation').reduce(ee.Reducer.stdDev()).float().rename("Elevation_STD");
var aspectmax =collection.select('aspect').reduce(ee.Reducer.max()).rename("aspect_Max");
var aspectmean = collection.select('aspect').reduce(ee.Reducer.mean()).rename("ASPECT");
var aspectstd = collection.select('aspect').reduce(ee.Reducer.stdDev()).float().rename("aspect_STD");
var hillshademax =collection.select('hillshade').reduce(ee.Reducer.max()).rename("hillshade_Max");
var hillshademean = collection.select('hillshade').reduce(ee.Reducer.mean()).rename("HILLSHADE");
var hillshadestd = collection.select('hillshade').reduce(ee.Reducer.stdDev()).float().rename("hillshade_STD");

//Add the index statistics to the median bands and clip the dataset with the ROI
var compclip = band_median
.addBands(ndvimean).addBands(ndvimax).addBands(ndvistd)
.addBands(ndwimean).addBands(ndwimax).addBands(ndwistd)
.addBands(savimean).addBands(savimax).addBands(savistd)
.addBands(ndmimean).addBands(ndmimax).addBands(ndmistd)
.addBands(evimean).addBands(evimax).addBands(evistd)
.addBands(slopemean)
.addBands(elevationmean)
.addBands(aspectmean)
.addBands(hillshademean).clip(roi);
print("Composition", compclip)

//Visualization of the final dataset using RGB and CIR
Map.addLayer(compclip, {min: 0.0, max: 0.3, bands: ['B4', 'B3','B2'],}, 'RGB');
Map.addLayer(compclip, {min: 0.0, max: 0.3, bands: ['B8', 'B4', 'B3'],}, 'CIR');

// Ensure all bands have compatible data types
var compclipExport = compclip.toDouble();

//China-Nepal Friendship Highway Definition
Map.addLayer(road,{color:'red'},'China-Nepal Friendship Highway');
Map.centerObject(road,9);

//**************************************************************************************************************************************
//Calculation of Feature Importance
//**************************************************************************************************************************************
var mergebands = band_median.addBands(ndvimean).addBands(ndwimean).addBands(savimean).addBands(ndmimean).addBands(evimean)
.addBands(slopemean).addBands(elevationmean).addBands(aspectmean).addBands(hillshademean).clip(roi);

var training = mergebands.sampleRegions({
  collection: landcover, 
  properties: ['SymbolID'], 
  scale: 10
});

// Train a classifier.
var classifier = ee.Classifier.smileRandomForest(50).train({
  features: training,  
  classProperty: 'SymbolID', 
  inputProperties: mergebands.bandNames()
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

// ***********************************************************************************************************************************************
// Image Export
// ***********************************************************************************************************************************************
// Export the image to an Earth Engine asset
Export.image.toAsset({
  image: compclip,
  region: roi,
  description: 'DATASET_2024',
  scale: 10
});

// Image Export
Export.image.toDrive({
  image:compclipExport,
  description:'CIR',
  region:roi,
  scale:10,
  maxPixels:1e13,
});