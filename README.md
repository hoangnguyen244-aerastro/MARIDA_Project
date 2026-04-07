# WAVELET-SVM FOR ANOMALY DETECTION IN SATELLITE IMAGERY

Project Structure



MARIDA\_Project: raw\_data, code, data, models, results, reports, README.md

=================================================================
HOW TO RUN ?

1. Open MATLAB in this directory
2. Run file "run\_all.m" in MATLAB r2015b-> Files in folder (data, models, results) will be generated.
If the file shows a path error
-> go to the "code" folder
-> open the "step1\_filter\_data.m" file
-> find this line:
root\_path = 'E:\\SpbSUITD\\Semester2\\Neural-Network\\MARIDA\_Project\\raw\_data\\MARIDA\\';
or
root\_path = '../raw\_data/MARIDA/';
And change the following line of code to your computer's path.
3. When you see files .mat and .xlsl in 3 folders.
Run script "plot\_results.m" -> Generate the image of plots into "reports" folder.

=================================================================
PIPELINE STEPS

1. Filter data (Normal: water only, Anomaly: ship/debris)
2. Extract 18 wavelet features (energy, entropy, skewness)
3. Train SVM with grid search (C, gamma)
4. Evaluate (accuracy, precision, recall, F1, AUC)
5. Compare wavelet families (db4, sym4, bior3.5, coif2, haar)

RESULTS

* Best wavelet: coif2
* Accuracy: \~66,7%

=================================================================

