# Disclaimer:
This was an old exercise. Therefore it might be possible that current approaches largely differ from this approach. Also keep in mind that this task was solved under potential restrictions (solution approaches, model usage, hardware, time).
# Pollution Prediction
We are given coordinates and need to predict pollution levels. We received a template for this exercise. The objective of this exercise was to try out Gaussian Processes and see how you can use them.

## Our Approach:
We use the maternal Kernel. Addiotionally we use Nystroem to create an approximate feature map. Furthermore, we adjust the lossfunction such that there is a different cost depending on the type of error.