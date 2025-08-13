import numpy as np
import matplotlib.pyplot as plt

class Hidden_Layer():
    """Creates a hidden layer object.
    
    Args: 
			input_dimension: an integer indicating the number of nodes the layer takes as input
      output_dimension: an integer indicating the number of nodes the layer outputs
    """
    def __init__(self,input_dimension,output_dimension):
        self.W = np.random.normal(0,1.0/np.sqrt(input_dimension), (input_dimension,output_dimension))
        self.b = np.zeros((1,output_dimension))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def linear_forward(self,x):
        self.yn_1 = x                       
        return np.dot(x,self.W) + self.b    
    
    def activation_forward(self,z):
        y = np.tanh(z)
        self.da = 1-y*y                    
        return y                            
    
    def backward(self,A_prev):
        
        Ada = np.multiply(A_prev,self.da)
        
        self.dW = np.matmul(self.yn_1.T,Ada)
        self.db = Ada.sum(axis=0)

        return np.matmul(Ada,self.W.T)
        
    def update(self,lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db


class Output_Layer():
    """Creates an output layer object.
    
    Args: 
			input_dimension: an integer indicating the number of nodes the layer takes as input
      output_dimension: an integer indicating the number of nodes the layer outputs
    """
    def __init__(self,input_dimension,output_dimension):
        self.W = np.random.normal(0,1.0/np.sqrt(input_dimension), (input_dimension,output_dimension))
        self.b = np.zeros((1,output_dimension))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def linear_forward(self,x):
        self.yn_1 = x                       
        return np.dot(x,self.W) + self.b    
    
    def activation_forward(self,z):        
        zmax = z.max(axis=1,keepdims=True) #prevents overflow for big z values
        expz = np.exp(z-zmax)
        Z = expz.sum(axis=1,keepdims=True)
        self.p = expz / Z                   
        return self.p

    
    def backward(self,labels):        
        p_of_f = self.p
        for i,lab in enumerate(labels):
            p_of_f[i,lab] -= 1
        
        self.db = p_of_f.sum(axis=0)
        self.dW = np.matmul(self.yn_1.T,p_of_f)

        return np.matmul(p_of_f,self.W.T)
        
    def update(self,lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db
        

class Network:
    """Creates a Network to concatenate layers.
    
    Args: 
			None
    """
    def __init__(self):
        self.layers = []
        
    def add(self,l):
        self.layers.append(l)
        
    def forward(self,y):
        for l in self.layers:          
            y = l.activation_forward(l.linear_forward(y))

        return y                 #returns p
    
    def backward(self,z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        #return z               #I don't need backward to return A(1)
    
    def update(self,lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)
                
class Perceptron():
	"""Creates a perceptron model object.

	The input layer must be the same size as the flattened image, and the output layer must be the same size as
	the number of classes. The hidden layers can be any size, i.e [input,h1,h2,...,output] 

	Args: 
		layer_nodes_list: a list of integers, containig the number of nodes of every layer.
	"""

	def __init__(self,layer_nodes_list):  
			self.model = Network()
			if len(layer_nodes_list) > 2:
					for i in range(len(layer_nodes_list)-2):
							self.model.add(Hidden_Layer(layer_nodes_list[i],layer_nodes_list[i+1]))
					self.model.add(Output_Layer(layer_nodes_list[-2],layer_nodes_list[-1]))
			elif len(layer_nodes_list) == 2:
					self.model.add(Output_Layer(layer_nodes_list[0],layer_nodes_list[1]))
			else:
					print("Specify at least the number of nodes of the input and output layers")
	
	def train(self,training_x,training_labels,learning_rate,batch_size,n_epochs=1): 
		"""Train the model for a given number of epochs epoch. 
    
    Args:
      training_x: training data in numpy array format
      training_labels: training labels in numpy array format
      learning_rate: learning rate
      batch_size: batch size
	  	n_epochs: number of epochs 
    """
		for n in range(n_epochs):
			for i in range(0,len(training_x),batch_size):
					xb = training_x[i:i+batch_size]
					yb = training_labels[i:i+batch_size]
					# forward pass
					p = self.model.forward(xb)
					# backward pass
					self.model.backward(yb)
					self.model.update(learning_rate)
	
	def accuracy(self,testing_x,testing_labels,conf_mat_toggle=0):  
		"""Test the accuracy of a given model and make a confusion matrix. 
    
 	 	Args:
    	testing_x: testing data in numpy array format
    	testing_labels: testing labels in numpy array format
      conf_mat_toggle: if 1 displays a confusion matrix and the most mislabeled pairs
	  
  	Returns: 
			The accuracy of the model as float
  	"""
		pred = np.argmax(self.model.forward(testing_x),axis=1)
		acc = (pred==testing_labels).mean()
		print(f"The accuracy of the model is {acc}")
		
		m = self.model.layers[-1].b.shape[1] #number of classes
		confusion_matrix = np.zeros((m,m))
		
		if conf_mat_toggle:
				for true_label,detected_label in zip(testing_labels,pred):
						if detected_label == true_label:
								confusion_matrix[int(detected_label),int(detected_label)] += 1
						else:
								confusion_matrix[int(detected_label),int(true_label)] += 1
				
				fig, ax = plt.subplots(1,1,figsize=(5,5)) 
				im=ax.matshow(confusion_matrix,aspect='auto')
				plt.title('Confusion matrix')
				plt.xlabel('true label')
				plt.ylabel('detected label')
				plt.show()
				max_conf = [0]
				conf_pairs = []
				N_max = 10
				N = confusion_matrix.sum()/100
				for i in range(m):
						for j in range(m):
								if i != j:
										if confusion_matrix[i,j]/N > max_conf[-1]:
												max_conf.append(confusion_matrix[i,j]/N)
												conf_pairs.append([i,j])
										elif confusion_matrix[i,j]/N > max_conf[0]:
												for n,k in enumerate(max_conf[1:]): 
														if confusion_matrix[i,j]/N < k:
																max_conf.insert(n+1,confusion_matrix[i,j]/N)
																conf_pairs.insert(n+1,[i,j])
																break
										if len(max_conf) > N_max:
												max_conf.pop(0)
												conf_pairs.pop(0)
												
				print('The',N_max,'most mislabeled pairs are:')
				for pair,perc in zip(list(reversed(conf_pairs)),list(reversed(max_conf))):
						print(pair,perc,'%')
				
		return acc
	
	
	def identify(self,digit):
			"""Identifies a digit
			
			Args: 
				digit: an image of a digit in numpy array format
			
			Returns:
                The digit the network identified
			"""
			p = self.model.forward(digit)
			return np.argmax(p)
	
	def clean(self):
			"""Deletes all the information stored in the model except for the weights, to save space when saving"""
			for l in range(len(self.model.layers[:-1])):
					for attr in ["dW","db","da","yn_1"]:
						try: 
							getattr(self.model.layers[l],attr)
							delattr(self.model.layers[l],attr)
						except AttributeError:
							pass
			for attr in ["dW","db","p","yn_1"]:
				try: 
						getattr(self.model.layers[-1],attr)
						delattr(self.model.layers[-1],attr)
				except AttributeError:
					pass

"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
#activate environment in VSCode
import utils.model_builder as modb
perc_2 = modb.Perceptron([784,20,20,10])
lr = 0.01
batch = 5
nepoch = 1
perc_2.train(training_data[0],training_data[1],lr,batch,nepoch)
perc_2.accuracy(test_data[0],test_data[1],1)"""