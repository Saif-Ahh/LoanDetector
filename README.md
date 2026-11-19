# Loan Detector

A neural network that predicts whether a borrower will pay back their loan or default on it, trained on LendingClub data. Everything is in one Colab notebook, `LoanDetector.ipynb`.

## Data

I used a subset (`lending_club_loan_two.csv`) of the [LendingClub dataset on Kaggle](https://www.kaggle.com/wordsforthewise/lending-club). It has about 396k loans and 27 columns, and each loan is labelled either "Fully Paid" or "Charged Off". The notebook also loads `lending_club_info.csv`, which just has a short description of each column.

Neither CSV is in the repo, so you'll need to grab them yourself.

## What the notebook does

### Exploring the data

- About 80% of the loans were fully paid, so the classes are pretty imbalanced
- `installment` and `loan_amnt` are almost perfectly correlated (0.95), which makes sense since one is calculated from the other
- Compared charge-offs across LendingClub's loan grades and sub-grades

### Cleaning

- Dropped `emp_title` (173k unique job titles, way too many to turn into dummies)
- Dropped `emp_length` because the charge-off rate is about the same no matter how long someone has been employed
- Dropped `title` since it's just a messier version of `purpose`
- Filled in missing `mort_acc` values with the average `mort_acc` for that row's `total_acc`, since those two were the most correlated
- Dropped the rows still missing `revol_util` or `pub_rec_bankruptcies` (less than 0.5% of the data)

### Features

- `term` converted to a number (36 or 60)
- Dummy variables for `sub_grade`, `verification_status`, `application_type`, `initial_list_status`, `purpose` and `home_ownership` (with NONE and ANY merged into OTHER)
- Pulled the zip code out of `address` and made dummies from that too
- Dropped `issue_d`, since you wouldn't know when a loan was issued before deciding whether to give it (data leakage)
- Kept only the year from `earliest_cr_line`

### Model

80/20 train/test split, with a MinMaxScaler fit on the training set only. The network is a Keras Sequential model:

```
Dense(78, relu)
Dropout(0.5)
Dense(39, relu)
Dropout(0.5)
Dense(19, relu)
Dense(1, sigmoid)
```

Trained with Adam and binary cross-entropy, batch size 256, for 50 epochs.

## Results

On the test set (79,044 loans):

- **89% accuracy**
- When it predicts a default, it's almost always right (6,720 out of 6,721)
- But it only catches about **43%** of the loans that actually defaulted (6,720 of 15,658)

So it plays it very safe. With 80% of loans being fully paid, it leans heavily toward predicting "paid", and 89% isn't a huge jump over just guessing "paid" every time. Next things I'd try are class weights and lowering the 0.5 cutoff to catch more of the defaults.

(In the notebook, the arguments to `confusion_matrix` and `classification_report` are in the wrong order, so precision and recall are flipped in that output. The numbers above are the corrected ones.)

## Running it

It was built in Google Colab, where pandas, numpy, matplotlib, seaborn, scikit-learn and TensorFlow are already installed.

1. Run the upload cell and pick `lending_club_info.csv`
2. Change the path in the `pd.read_csv(...)` cell to wherever you saved `lending_club_loan_two.csv` (mine was on Google Drive)
3. Run everything else top to bottom

To run it locally instead, swap the Colab upload and Drive cells for a normal `pd.read_csv` call.
