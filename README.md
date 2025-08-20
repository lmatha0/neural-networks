neural networks:

protein classification:
- A feed forward neural network to showcase the concepts used in a simple neural network system. implemented a 2 layer hidden network architecture to understand forward propagation and backward propagation between each layer
- used to classify proteins based on parameters including resolution, pH, molecular weight, density and more
- created from scratch using pandas to load data and numpy to perform calculations. also used scikitlearn to normalize and split data into train/test sets
- dataset was very large and difficult to clean, resulting in an accuracy of about 58% and a loss that was slightly increasing
- will extend to a neural network with an arbitrary number of hidden networks using another dataset

protein deeplayer:
- similar to the above project, but extended to an n number of layers defined by the user
- used to classify proteins on 5 different structures (structural, transport, enzyme, receptor, other)
- completed task in pandas to load dataset and numpy to perform calculations
- note that another model achieved a similar accuracy of 20% (https://www.kaggle.com/code/t8101349/bioinformatics-protein-dataset-analysis)

RNN_Text_Generation:
- a basic rnn that was used to generate text based on sequences of a sentence
- explores concepts used in an rnn such as calculating the activations of each cell and propagating it forward. also calculated backprop to update gradients and sample text
- used numpy to perform vectorize words and perform calculations for each rnn cell
- text taken from (https://www.gutenberg.org/files/11/11-0.txt)
