# Fetch Receipt Count
to do: Mention assumptions and notes, Mention evaluation metric and sanity checks
Notes and Assumptions:
1. The receipt counts in the training set were grouped by the month number because of two reasons: a) The data was fluctuating constantly but the overall trend was positive and the growth rate was quite consistent, and b) We were only going to predict the receipt counts for the entire months in the testing time. Rather than aggregating later, I aggregated early and made the training set simple.
2. All X values were divided by 1000000 (1e6), which means all the training and predicted values for receipt counts are in millions.
3. Even though the training loss curve shows that there was not a lot of decrease in the loss for the last couple 1000 epochs, I felt it was worth spending a couple of extra seconds during training for slight increase in the accuracy of the results. Overfitting was not really a concern here, we had 11 training points and we would want the line to overfit to the data as much as possible in this case.

Sanity Check for Results:
1. Since I implemented linear regression in Pytorch, verifying the results from the numpy's polyfit funtion seemed like a decent move. Polyfit function of the numpy library estimates the best-fit line for the given X and y values, and gives the parameters m (slope) and c (intercept) of the equation y = mX + c (when degree = 1). The polyfit function estimated the equation as: y = 7.17*X + 221.88. My model estimated it as y = 7.81x + 216.63, which is pretty close. 
2.
