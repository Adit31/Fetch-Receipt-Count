# Fetch Receipt Count
Please go to the Github link to access the codebase: https://github.com/Adit31/Fetch-Receipt-Count/tree/main

Notes and Assumptions:
1. The receipt counts in the training set were grouped by the month number because of two reasons: a) The data was fluctuating constantly but the overall trend was positive and the growth rate was quite consistent, and b) We were only going to predict the receipt counts for the entire months in the testing time. Rather than aggregating later, I aggregated early and made the training set simple.
2. All X values were divided by 1000000 (1e6), which means all the training and predicted values for receipt counts are in millions.
3. Even though the training loss curve shows that there was not a lot of decrease in the loss for the last couple 1000 epochs, I felt it was worth spending a couple of extra seconds during training for slight increase in the accuracy of the results. Overfitting was not really a concern here, we had 11 training points and we would want the line to overfit to the data as much as possible in this case.
4. I tried out a fully connected neural net with 2 or 3 layers as well, but as simple as linear regression is, it gave the best results. Hence, I decided to stick with it. Using any other architecture did not make a lot of sense to me in this case.

Sanity Check for Results and Evaluation Metric:
1. Since I implemented linear regression in Pytorch, verifying the results from the numpy's polyfit funtion seemed like a decent move. Polyfit function of the numpy library estimates the best-fit line for the given X and y values, and gives the parameters m (slope) and c (intercept) of the equation y = mX + c (when degree = 1). The polyfit function estimated the equation as: y = 7.17*X + 221.88. My model estimated it as y = 7.81x + 216.63, which is pretty close. 
2. Growth rate should be consistent over 2021 and 2022 because of the linear correlation. The approximate difference between December 2021 and January 2021 is almost similar to the difference between December 2022 and January 2022, i.e., ~80 Million. This indicates that the model is producing decent results.
3. MSE loss was used during the training and testing of the model, which showed a declining trend, as it should.

Steps to run the Application:
1. The app is hosted online and can be directly accessed by clicking on the url: https://fetch-receipt-count.streamlit.app/
2. If you'd like to build the application locally and then launch it, the container image has been uploaded on docker hub and can be pulled to the local system using the command: docker pull adit31/fetch-app:latest
3. After pulling the docker image to your repository, run the container by the command: docker run -p 8501:8501 fetch-app
4. It might take some time to pull the docker image and run it because the file size is quite big, i.e., ~8 GB.
