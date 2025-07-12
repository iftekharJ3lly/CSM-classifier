

The model achieves ~82% training accuracy, but test accuracy plateaus around 63–66%, indicating overfitting.
* Training loss decreases steadily, while **test loss** fluctuates and begins to increase after \~20 epochs, further confirming overfitting.
* A 15–20% accuracy gap suggests the model is learning training data well but struggles to generalize to unseen data.
* Possible reasons include:

  * Noisy or imbalanced data (e.g., AI-generated images in Aki training data)
  * Limited data diversity across classes
* Future improvements could include:

  * Adding dropout and weight decay
  * Implementing early stopping
  * Applying data augmentation for better generalization
  * Using a pretrained backbone (e.g., ResNet or MobileNet)


