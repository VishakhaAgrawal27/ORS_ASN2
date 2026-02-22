// ============================================
// ASSIGNMENT 2 – CLOUD REMOVAL DATA EXTRACTION
// ============================================

// ============================================
// ██████  CONFIG — EDIT ONLY THIS SECTION
// ============================================

var STUDY_AREA_ASSET = 'projects/my-coral-project/assets/geo';
var STUDY_AREA_FIELD = 'shapeName';
var STUDY_AREA_VALUE = 'Sangareddy';

var CLOUDY_PERIODS = [
  { label: 'oct2024', start: '2024-10-01', end: '2024-10-30', minCloud: 30, maxCloud: 85 },
  { label: 'aug2024', start: '2024-08-01', end: '2024-08-31', minCloud: 30, maxCloud: 85 },
  { label: 'sep2024', start: '2024-09-01', end: '2024-09-30', minCloud: 30, maxCloud: 85 },
];

var CLEAN_START        = '2024-11-01';
var CLEAN_END          = '2024-12-31';
var CLEAN_MAX_CLOUD    = 10;

var EXPORT_FOLDER      = 'anagha_data_scl_asn2';
var EXPORT_SCALE_BANDS = 10;
var EXPORT_SCALE_SCL   = 20;
var EXPORT_PREFIX      = 'sangareddy';
var BANDS              = ['B2', 'B3', 'B4', 'B8'];

// Complementarity threshold — what % of the study area
// must be clear in at least one image for the set to be usable
var COMPLEMENTARITY_THRESHOLD = 80; // percent

// ============================================
// ██████  END CONFIG — DO NOT EDIT BELOW
// ============================================


// ----------------------------
// Load study area
// ----------------------------
var districts = ee.FeatureCollection(STUDY_AREA_ASSET);
var studyArea = districts.filter(ee.Filter.eq(STUDY_AREA_FIELD, STUDY_AREA_VALUE));
var roi = studyArea.geometry();

Map.centerObject(studyArea, 9);
Map.addLayer(
  studyArea.style({ color: 'red', width: 2, fillColor: '00000000' }),
  {}, STUDY_AREA_VALUE + ' Boundary'
);


// ----------------------------
// Cloud mask function (QA60)
// ----------------------------
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0)
               .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask);
}


// ----------------------------
// SCL-based cloud mask per image
// Returns 1 = cloudy, 0 = clear
// ----------------------------
function getSCLCloudMask(image) {
  var scl = image.select('SCL');
  // SCL classes: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
  return scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10))
            .rename('cloud_mask');
}


// ----------------------------
// Loop over cloudy periods
// ----------------------------
var cloudyImages   = {};   // band images
var cloudMasks     = {};   // per-image cloud masks for complementarity
var clearMasks     = {};   // per-image clear masks (inverse)

CLOUDY_PERIODS.forEach(function(p) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea)
    .filterDate(p.start, p.end)
    .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', p.minCloud))
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', p.maxCloud));

  print(p.label + ' scenes (' + p.minCloud + '–' + p.maxCloud + '% cloud):', col.size());

  var img = col.mosaic().clip(roi);

  // --- Cloud mask from SCL for this mosaic ---
  var sclMosaic   = col.select('SCL').mosaic().clip(roi);
  var cloudMask   = getSCLCloudMask(sclMosaic);          // 1 = cloud
  var clearMask   = cloudMask.eq(0).rename('clear');     // 1 = clear

  cloudMasks[p.label] = cloudMask;
  clearMasks[p.label] = clearMask;

  // --- Per-image cloud % over ROI ---
  var cloudPct = cloudMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 20,
    maxPixels: 1e10
  });
  print(p.label + ' cloud % over ROI (SCL-based):',
        ee.Number(cloudPct.get('cloud_mask')).multiply(100).round());

  var bandImg = img.select(BANDS);
  cloudyImages[p.label] = bandImg;

  Map.addLayer(bandImg, { bands: ['B4','B3','B2'], min: 0, max: 3000 }, 'Cloudy ' + p.label);

  // Visualise cloud mask per image
  Map.addLayer(cloudMask.selfMask(),
    { palette: ['white'] }, p.label + ' cloud mask', false);

  Export.image.toDrive({
    image: bandImg,
    description: EXPORT_PREFIX + '_' + p.label + '_B2B3B4B8_' + EXPORT_SCALE_BANDS + 'm',
    folder: EXPORT_FOLDER,
    region: roi,
    scale: EXPORT_SCALE_BANDS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
});


// ============================================
// SPATIAL COMPLEMENTARITY CHECK
// ============================================
//
// For each pixel, check if it is clear in AT LEAST ONE
// of the 3 cloudy inputs. If yes → PMAA can reconstruct it.
// If no → permanently clouded across all inputs → unrecoverable.
//
// covered pixel = clear in img1 OR clear in img2 OR clear in img3
// ============================================

var labels = CLOUDY_PERIODS.map(function(p) { return p.label; });

// Union of all clear masks: pixel = 1 if clear in ANY input
var unionClear = ee.Image(0);
labels.forEach(function(lbl) {
  unionClear = unionClear.or(clearMasks[lbl]);
});
unionClear = unionClear.rename('covered').clip(roi);

// % of study area covered by at least one clear observation
var coverageStats = unionClear.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 20,
  maxPixels: 1e10
});
var coveragePct = ee.Number(coverageStats.get('covered')).multiply(100).round();

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('SPATIAL COMPLEMENTARITY REPORT');
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('% of ROI clear in at least 1 input:', coveragePct);
print('Threshold for usable set:', COMPLEMENTARITY_THRESHOLD + '%');
print('PASS if coverage >= threshold');

// Per-image individual clear coverage
labels.forEach(function(lbl) {
  var s = clearMasks[lbl].reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 20,
    maxPixels: 1e10
  });
  print(lbl + ' clear % over ROI:',
        ee.Number(s.get('clear')).multiply(100).round());
});

// Pairwise overlap check — how much do cloud-free zones complement each other?
// i.e. pixel clear in img1 but cloudy in img2 = img1 uniquely contributes there
for (var i = 0; i < labels.length; i++) {
  for (var j = i + 1; j < labels.length; j++) {
    var lbl1 = labels[i], lbl2 = labels[j];
    // Pixels clear in BOTH = redundant overlap (both see it clearly)
    var bothClear = clearMasks[lbl1].and(clearMasks[lbl2]);
    var overlapStats = bothClear.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 20,
      maxPixels: 1e10
    });
    print(lbl1 + ' ∩ ' + lbl2 + ' both-clear overlap %:',
          ee.Number(overlapStats.get('clear')).multiply(100).round());
  }
}

// --- Visualise uncovered pixels (never clear in any input) ---
var uncovered = unionClear.eq(0).selfMask().rename('uncovered');

Map.addLayer(unionClear.selfMask(),
  { palette: ['00cc44'] }, 'Complementarity: covered (≥1 clear input)', false);

Map.addLayer(uncovered,
  { palette: ['ff0000'] }, 'Complementarity: UNCOVERED (all inputs cloudy)', true);

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('Red pixels on map = permanently clouded across all 3 inputs.');
print('PMAA cannot recover these without additional input dates.');
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

// Export the coverage map for reference
Export.image.toDrive({
  image: unionClear.toByte(),
  description: EXPORT_PREFIX + '_complementarity_coverage_map',
  folder: EXPORT_FOLDER,
  region: roi,
  scale: 20,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});


// ----------------------------
// Clean reference image
// ----------------------------
var cleanCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate(CLEAN_START, CLEAN_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLEAN_MAX_CLOUD))
  .map(maskS2clouds);

print('Clean scenes (' + CLEAN_START + ' → ' + CLEAN_END +
      ', <' + CLEAN_MAX_CLOUD + '% cloud):', cleanCollection.size());

var clean = cleanCollection.median().select(BANDS).clip(roi);
Map.addLayer(clean, { bands: ['B4','B3','B2'], min: 0, max: 3000 }, 'Clean Reference');

var cleanLabel = 'novdec2024med';

Export.image.toDrive({
  image: clean,
  description: EXPORT_PREFIX + '_' + cleanLabel + '_B2B3B4B8_' + EXPORT_SCALE_BANDS + 'm',
  folder: EXPORT_FOLDER,
  region: roi,
  scale: EXPORT_SCALE_BANDS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});


// ----------------------------
// SCL export
// ----------------------------
var sclClean = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate(CLEAN_START, CLEAN_END)
  .select('SCL')
  .mode()
  .clip(roi);

Map.addLayer(sclClean, {
  min: 0, max: 11,
  palette: ['000000','ff0000','404040','a06000','00ff00','ffff00',
            '0000ff','808080','c0c0c0','ffffff','00ffff','ff00ff']
}, 'SCL Clean Ref');

Export.image.toDrive({
  image: sclClean,
  description: EXPORT_PREFIX + '_' + cleanLabel + '_SCL_' + EXPORT_SCALE_SCL + 'm',
  folder: EXPORT_FOLDER,
  region: roi,
  scale: EXPORT_SCALE_SCL,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('All exports queued. Check Tasks tab.');
