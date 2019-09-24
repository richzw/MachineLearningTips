-- It's true that a Variable can be used any place a Tensor can, 

but the key differences between the two are that a Variable maintains its state across multiple calls to run() and 
a variable's value can be updated by backpropagation (it can also be saved, restored etc as per the documentation).

These differences mean that you should think of a variable as representing your model's trainable parameters 
(for example, the weights and biases of a neural network), while you can think of a Tensor as representing 
the data being fed into your model and the intermediate representations of that data as it passes through your model.

--- another view point

-- Tensors and variables serve different purposes. 

Tensors (tf.Tensor objects) can represent complex compositions of mathematical expressions, like loss functions in a neural network, 
or symbolic gradients. 

Variables represent state that is updated over time, like weight matrices and convolutional filters during training. 

While in principle you could represent the evolving state of a model without variables, you would end up with a very large 
(and repetetive) mathematical expression, so variables provide a convenient way to materialize the state of the model, 
and—for example—share it with other machines for parallel training.
