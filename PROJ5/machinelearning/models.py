import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)
        self.batch_size = 1

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # print(self)
        # print("ASDA")
        # print(x)
        weights = self.get_weights()
        node = nn.DotProduct(x,weights)
        return node 

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        runNode = self.run(x)
        answScalar = nn.as_scalar(runNode)
        if answScalar < 0 :
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        
        while True:
            converged = True
            for x, y in dataset.iterate_once(self.batch_size):
                prediciton = self.get_prediction(x)
                yScalar = nn.as_scalar(y)
                if prediciton != yScalar:
                    nn.Parameter.update(self.w,x,yScalar)
                    converged = False
            if converged:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 20
        self.firstLayerWeight = nn.Parameter(1,100)
        self.secondLayerWeigth = nn.Parameter(100,1)
        self.firstLayerBs = nn.Parameter(1,100)
        self.secondLayerBs = nn.Parameter(1,1)
        self.learning_rate = -0.05


    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        firstl = nn.Linear(x,self.firstLayerWeight)
        # print("ASDA----------------------------ASDAs")
        # print(firstl)
        firstPlusBias = nn.AddBias(firstl,self.firstLayerBs)

        # print("ASDA----------------------------ASDAs")
        # print(firstPlusBias)
        firstRelu = nn.ReLU(firstPlusBias)
        secondLayer = nn.Linear(firstRelu,self.secondLayerWeigth)
        secondLayerPlusBias = nn.AddBias(secondLayer , self.secondLayerBs)
        # print("ASDA--asdkapskjdoa--------------------------ASDAs")
        # print(secondLayerPlusBias)
        return secondLayerPlusBias


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = 0 
    
        while True:
            # x,y = nn.Constant(data.x),nn.Constant(data.y)
            # print(x,y)
            for x ,y in data.iterate_once(self.batch_size):
                nonScalarLoss = self.get_loss(x,y)
                loss = nn.as_scalar(nonScalarLoss)
               
                if loss <= 0.009:
                    return 
            
                gradw1,gradw2,gradb1,gradb2 = self.getGradient(nonScalarLoss)
                self.updateParams(gradw1,gradw2,gradb1,gradb2)

    def updateParams(self,gradw1,gradw2,gradb1,gradb2):
        self.firstLayerWeight.update(gradw1,self.learning_rate)
        self.secondLayerWeigth.update(gradw2,self.learning_rate)
        self.firstLayerBs.update(gradb1,self.learning_rate)
        self.secondLayerBs.update(gradb2,self.learning_rate)

    def getGradient(self,loss):
        return nn.gradients(loss,[self.firstLayerWeight,self.secondLayerWeigth,self.firstLayerBs,self.secondLayerBs])


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.initSize()
        self.createMatrix()
    
    def initSize(self):
        self.digitSizepixel = 28
        self.VectorDimension = 784
        self.outputVectorSize = 10
        self.batch_size = 200
        self.MatrixDimension = 90
    
    def createMatrix(self):

        self.firstLayerWeight = nn.Parameter(self.VectorDimension,self.MatrixDimension)
        self.secondLayerWeigth = nn.Parameter(self.MatrixDimension,self.outputVectorSize)
        self.firstLayerBs = nn.Parameter(1,self.MatrixDimension)
        self.secondLayerBs = nn.Parameter(1,self.outputVectorSize)
        self.learning_rate = -0.6

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        firstl = nn.Linear(x,self.firstLayerWeight)
        # print(firstl)
        firstPlusBias = nn.AddBias(firstl,self.firstLayerBs)
        # print(firstPlusBias)
        firstRelu = nn.ReLU(firstPlusBias)
        secondLayer = nn.Linear(firstRelu,self.secondLayerWeigth)
        secondLayerPlusBias = nn.AddBias(secondLayer , self.secondLayerBs)
        # print(secondLayerPlusBias)
        return secondLayerPlusBias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return  nn.SoftmaxLoss(self.run(x),y)

    def train(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0 
        count = 0 
        while True:
            print("GENERATION",count)
            count += 1
            for x, y in data.iterate_once(self.batch_size):
            # x,y = nn.Constant(data.x),nn.Constant(data.y)
                nonScalarLoss = self.get_loss(x,y)
                accuracy = data.get_validation_accuracy()
                # print(accuracy) #TODO DELETE THISx
                if accuracy > 0.975:
                    return
                gradw1,gradw2,gradb1,gradb2 = self.getGradient(nonScalarLoss)
                self.updateParams(gradw1,gradw2,gradb1,gradb2)
            print("ACCURACY",accuracy)

    def updateParams(self,gradw1,gradw2,gradb1,gradb2):
        self.firstLayerWeight.update(gradw1,self.learning_rate)
        self.secondLayerWeigth.update(gradw2,self.learning_rate)
        self.firstLayerBs.update(gradb1,self.learning_rate)
        self.secondLayerBs.update(gradb2,self.learning_rate)

    def getGradient(self,loss):
        return nn.gradients(loss,[self.firstLayerWeight,self.secondLayerWeigth,self.firstLayerBs,self.secondLayerBs])

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.initConstants()
        self.initMatrix()

    def initConstants(self):
        self.MatrixSize = 150
        self.outSize= 5
        self.batch_size = 80
        self.learning_rate = -0.08


    def initMatrix(self):
        self.firstLayerWeight = nn.Parameter(self.num_chars, self.MatrixSize)
        self.firstLayerBs = nn.Parameter(1,self.MatrixSize)
        self.secondLayerWeigth = nn.Parameter(self.MatrixSize,self.MatrixSize)
        self.secondLayerBs = nn.Parameter(1,self.MatrixSize)
        self.outPutWeights = nn.Parameter(self.MatrixSize,self.outSize)
        self.outBs = nn.Parameter(1,self.outSize)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        answ = None
        for i in range(len(xs)):
            if i == 0:
                answ = self.processInital(xs,i)
            else:
                answ = self.processNormal(xs,i,answ)
        linear = nn.Linear(answ,self.outPutWeights)
        bias = nn.AddBias(linear,self.outBs)
        return bias
            
    def  processInital(self,xs,i):
        linear = nn.Linear(xs[i],self.firstLayerWeight)
        bias = nn.AddBias(linear,self.firstLayerBs)
        relu = nn.ReLU(bias)
        return relu
        
    def processNormal(self,xs,i,answ):
        linear = nn.Linear(xs[i],self.firstLayerWeight)
        bias = nn.AddBias(linear,self.firstLayerBs)
        lin1 = nn.Linear(answ,self.secondLayerWeigth)
        bias1 = nn.AddBias(lin1,self.secondLayerBs)
        return nn.ReLU(nn.Add(bias1,bias))
        


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, data):
        """
        -FInish
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        accuracy = 0 
        count = 0 
        while True:
            print("GENERATION",count)
            count += 1
            for x, y in data.iterate_once(self.batch_size):
            # x,y = nn.Constant(data.x),nn.Constant(data.y)
                nonScalarLoss = self.get_loss(x,y)
                accuracy = data.get_validation_accuracy()
                # print(accuracy) #TODO DELETE THISx
                if accuracy > 0.889:
                    return
                gradw1,gradw2,gradw3,gradb1,gradb2,gradb3 = self.getGradient(nonScalarLoss)
                self.updateParams(gradw1,gradw2,gradw3,gradb1,gradb2,gradb3)
            print("ACCURACY",accuracy)

    def updateParams(self,gradw1,gradw2,gradw3,gradb1,gradb2,gradb3):
        self.firstLayerWeight.update(gradw1,self.learning_rate)
        self.secondLayerWeigth.update(gradw2,self.learning_rate)
        self.outPutWeights.update(gradw3,self.learning_rate)
        self.firstLayerBs.update(gradb1,self.learning_rate)
        self.secondLayerBs.update(gradb2,self.learning_rate)
        self.outBs.update(gradb3,self.learning_rate)

    
    def getGradient(self,loss):
        return nn.gradients(loss,[self.firstLayerWeight,self.secondLayerWeigth,self.outPutWeights, self.firstLayerBs,self.secondLayerBs, self.outBs])