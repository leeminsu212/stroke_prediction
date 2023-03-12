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
Use three classifier(Decision tree classifier, Random forest classifier, K-neighbors classifier). 

