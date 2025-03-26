# CS50 AI Shopping

Shopping project builds a k-nearest neighbours classifier to predict if a customer will complete a purchase based on their browsing behavior, such as time spent on pages and products viewed. The classifier outputs a purchase likelihood prediction for each user.

## Contributions

`shopping.py`:

`load_data`: Reads a CSV file and returns two lists--evidence (features for each data point) and labels (whether a purchase was made). Converts columns to appropriate data types and maps categorical data to numeric values.

`train_model`: Accepts evidence and labels, then trains and returns a k-nearest neighbours classifier using scikit-learn’s KNeighborsClassifier.

`evaluate`: Takes true labels and predicted labels as input, and calculates the sensitivity (true positive rate) and specificity (true negative rate) to evaluate the classifier's performance.

### Testing

A test script (`test_shopping.py`) has been developed to verify the correct operation of all listed functions.

### Technologies Used

- `Unittest`
- `scikit-learn`

### Usage

- main: `python3 shopping.py data`
- test: `python3 test_shopping.py`