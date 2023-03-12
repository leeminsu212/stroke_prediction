# Stroke prediction

## 1. Objective setting
Try to identify the cause of stroke and prevent it through data analysis.</br>
Dataset from Kaggle https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## 2. EDA
<img src="https://user-images.githubusercontent.com/33173280/223042237-eae0584d-a9ad-43f0-9a3f-b4e6a21459cd.png" width="400" height="500">
This dataset have 12 features(7 numerical and 5 categorical) and 5110 rows.</br>
There are some null value in bmi column. We droppd these columns.
</br></br></br>

<div>
<img src="https://user-images.githubusercontent.com/33173280/224318912-52c03b5d-2c22-40cb-9a9d-3af149545555.png" width="400" height="400">
<img src="https://user-images.githubusercontent.com/33173280/224319198-a46cc75b-92eb-4a97-ac5c-e2379fe1c1ed.png" width="400" height="400">
</div>
Large proportion of data for people without stroke. &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp There is wrong value in gender column.</br></br></br>

<div>
<img src="https://user-images.githubusercontent.com/33173280/224352010-57468e1b-e20f-43e8-aab3-b3c9128db9e7.png" width="300" height="500">
<img src="https://user-images.githubusercontent.com/33173280/224349406-4bf2b5c6-64a8-440d-b91a-dc27fee9eee1.png" width="520" height="500">
</div>
Use the SelectKBest method from scikit-learn to determine the correlation between each column and the target column(stroke). And print out the heatmap for some of top columns.
</br></br></br>

<img src="https://user-images.githubusercontent.com/33173280/224523358-02d14ded-fbbd-4e47-8262-74b52d13f928.png" width="820" height="350">
Boxplot showing outlier in age column and avg_glucose_level column.
</br></br></br>

<img src="https://user-images.githubusercontent.com/33173280/224523924-2068fe2a-90cd-4c65-90de-cb8bf76247bf.png" width="820" height="400">
Comparison of age groups between people with and without stroke. Most of the people with stroke are older people.

## 3. Data analysis
<img src="https://user-images.githubusercontent.com/33173280/224543810-0dbce6cf-79b3-466a-967e-35e725079bd5.png" width="700" height="400">

Use three classifiers(**Decision tree classifier**, **Random forest classifier**, **K-neighbors classifier**) for stroke prediction.</br>
For each classifier, make combinations using 2 encoders(**Label encoder**, **OneHot encoder**) and 4 scalers(**Standard scaler**, **Robust scaler**, **MaxAbs scaler**, **MinMax scaler**). Then compare them to choose the best encoder and scaler for each classifier. 
</br></br></br>

<img src="https://user-images.githubusercontent.com/33173280/224544832-e9d94a86-13bc-4f18-a7c8-44f4f5e24b93.png" width="700" height="450">
<img src="https://user-images.githubusercontent.com/33173280/224544907-f52ea9d9-eed4-4e0e-8812-410314f9f73b.png" width="700" height="450">
<img src="https://user-images.githubusercontent.com/33173280/224544921-220f6e49-bd98-4cd0-ab7a-fd8464d6946e.png" width="700" height="450">

Use K-fold validation(k=5) to compare accuracy of each model.</br>
K-neighbors classifier's accuracy is the best(0.949 on test set). However, looking at the confusion matrix, it did not predict correctly for data that stroke=1(with stroke). I think it is because of the number of data with stroke is too small compared to the number of data without stroke.
