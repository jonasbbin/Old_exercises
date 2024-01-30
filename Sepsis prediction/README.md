# Disclaimer:
This was an old exercise. Therefore it might be possible that current approaches largely differ from this approach. Also keep in mind that this task was solved under potential restrictions (solution approaches, model usage, hardware, time).
# Sepsis Prediction
We are given a table full of medical measurements and need to solve different tasks with it. We need to predict different labels/conditions like sepsis, heartrate etc. The aim of the task is to perform data preprocessing and feature selection to improve a models performance.

## Our Approach:
We do manual data preprocessing and hard coded feature selection. Depending on the task we either use SVC or SVR to predict the labels.