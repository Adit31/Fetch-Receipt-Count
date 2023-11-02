# Fetch Receipt Count
to do: Mention assumptions and notes, Mention evaluation metric and sanity checks
Notes and Assumptions:
1. The receipt counts in the training set were grouped by the month number because of two reasons: a) The data was fluctuating constantly but the overall trend was positive and the growth rate was quite consistent, and b) We were only going to predict the receipt counts for the entire months in the testing time. Rather than aggregating later, I aggregated early and made the training set simple.
2. All X values were divided by 1000000 (1e6), which means all the training and predicted values for receipt counts are in millions.

Sanity Check for Results:
1. Since I implemented linear regression in Pytorch, verifying the results from the numpy's polyfit funtion seemed like a decent move. Polyfit function of the numpy library estimates the best-fit line for the given X and y values, and gives the parameters m (slope) and c (intercept) of the equation y = mX + c (when degree = 1). The polyfit function estimated the parameters as: m = 7.1688581713286625 and c = 221.8771393030302. 
