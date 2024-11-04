import os
# datasets = ['ECGFiveDays','Fungi','CBF','BME','UMD','DiatomSizeReduction','DodgerLoopWeekend','DodgerLoopGame','GunPoint','Coffee','FaceFour','FreezerSmallTrain','ArrowHead']
datasets = ['Meat','Trace','MelbournePedestrian','MiddlePhalanxTW','DistalPhalanxOutlineAgeGroup','MiddlePhalanxOutlineAgeGroup','ProximalPhalanxTW','ProximalPhalanxOutlineAgeGroup','DistalPhalanxTW','Herring','Car','InsectEPGRegularTrain','MedicalImages','Lightning2','FreezerRegularTrain','Ham','MiddlePhalanxOutlineCorrect','DistalPhalanxOutlineCorrect','ProximalPhalanxOutlineCorrect','Mallat','InsectWingbeatSound','Rock','GesturePebbleZ1','SwedishLeaf','CinCECGTorso','GesturePebbleZ2','Adiac','ECG5000','WordSynonyms','FaceAll','GestureMidAirD2','GestureMidAirD3','GestureMidAirD1','ChlorineConcentration','HouseTwenty','Fish','OSULeaf','MixedShapesSmallTrain','CricketZ','CricketX','CricketY','FiftyWords','Yoga','TwoPatterns','PhalangesOutlinesCorrect','Strawberry','ACSF1','AllGestureWiimoteY','AllGestureWiimoteX','AllGestureWiimoteZ','Wafer','WormsTwoClass','Worms','Earthquakes','Haptics','Computers','InlineSkate','PigArtPressure','PigCVP','PigAirwayPressure','Phoneme','ScreenType','LargeKitchenAppliances','SmallKitchenAppliances','RefrigerationDevices','UWaveGestureLibraryZ','UWaveGestureLibraryY','UWaveGestureLibraryX','ShapesAll','Crop','SemgHandGenderCh2','EOGVerticalSignal','EOGHorizontalSignal','MixedShapesRegularTrain','SemgHandMovementCh2','SemgHandSubjectCh2','PLAID','UWaveGestureLibraryAll','ElectricDevices','EthanolLevel','StarLightCurves','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','FordA','FordB','HandOutlines']

augs = ['cutout','jitter','scaling','time_warp','magnitude_warp','window_slice','window_warp','subsequence']
ps = [0.5]
cmd = 'python -u train.py {} UCR --archive UCR --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval --a1 {} --a2 {} --aug_p {} --log_file {}'
log_file = 'tuning'
for d in datasets:
    for i in range(len(augs)):
        a1 = augs[i]
        for j in range(i,len(augs)):
            a2 = augs[j]
            for p in ps:
                t = cmd.format(d,a1,a2,p,log_file)
                os.system(t)
