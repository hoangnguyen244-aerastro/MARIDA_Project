# WAVELET-SVM FOR ANOMALY DETECTION IN SATELLITE IMAGERY

Project Structure

MARIDA\_Project/
├── README.md (Explain about the processing flow/ Gthích luồng xử lý)
│
├── raw\_data/ (MARIDA set download(input)/Bộ MARIDA tải về-đầu vào)
│
├── code/ (Contains .m files/ Chứa các file .m)
│   ├── step1\_filter\_data.m
│   ├── step2\_extract\_features.m
│   ├── step3\_train\_svm.m
│   ├── step4\_evaluate.m
│   ├── step5\_compare\_wavelets.m
│   ├── wavelet\_feature\_extractor.m
│   ├── run\_all.m

│   └── plot\_results.m

│
├── data/(Contains processed data/Chứa dữ liệu đã xử lý)
│   ├── file\_lists.mat
│   └── features\_data.mat
│
├── models/(Contains the trained model/Chứa mô hình đã train)
│   └── svm\_model.mat
│
├── results/(Contains evaluation results/Chứa kết quả đánh giá)
│   ├── evaluation\_results.mat
│   └── wavelet\_comparison.mat
│   └── wavelet\_comparison.xlsl
│
└── reports/(Contains figures and tables for the article/Chứa figures, bảng biểu cho báo)
├── wavelet\_comparison.png (so sánh 5 wavelet)
├── confusion\_matrix.png (ma trận nhầm lẫn)
├── feature\_importance.png (top 10 đặc trưng quan trọng nhất)
└── roc\_curve.png (ROC curve với gtri AUC)

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

* Best wavelet: db4
* Accuracy: \~88-90%
* AUC-ROC: \~0.92

=================================================================

