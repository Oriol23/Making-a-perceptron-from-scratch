import numpy as np
def train_epoch(Network, training_x, training_labels, lr, batch_size: int) -> None:
    """Train a given model for a single epoch. 
    
    Args:
			Network: A Network class object, containig at least one layer
      training_x: training data in numpy array format
      training_labels: training labels in numpy array format
      lr: learning rate
      batch_size: batch size
      """
    for i in range(0,len(training_x),batch_size):
      xb = training_x[i:i+batch_size]
      yb = training_labels[i:i+batch_size]
			# forward pass
      p = Network.forward(xb)
			# backward pass
      Network.backward(yb)
      Network.update(lr)
      
def test_model(Network, testing_x, testing_labels):
  """Test the accuracy of a given model. 
    
  Args:
		Network: A Network class object, containig at least one layer
    testing_x: testing data in numpy array format
    testing_labels: testing labels in numpy array format
      
  Returns: 
		The accuracy of the model as float
  """
  pred = np.argmax(Network.forward(testing_x),axis=1)
  acc = (pred==testing_labels).mean()
  print(f"The accuracy of the model is {acc}")
  return acc