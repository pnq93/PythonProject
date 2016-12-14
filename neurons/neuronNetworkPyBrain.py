from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

network = FeedForwardNetwork() # create network
inputLayer = SigmoidLayer(1) # maybe LinearLayer ?
hiddenLayer = SigmoidLayer(4)
outputLayer = SigmoidLayer(1) # maybe LinearLayer ?

network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)
# Connection
network.addConnection(FullConnection(inputLayer, hiddenLayer))
network.addConnection(FullConnection(hiddenLayer, outputLayer))

network.sortModules()

dataTrain = SupervisedDataSet(1, 1) # input, target
dataTrain.addSample(1,0.76) # it seems to me that input(our x), target(value y) from function sin(x)*sin(2*x)


trainer = BackpropTrainer(network, dataTrain) # it's back prop, we use our network and our data
print(trainer.train()) # i think it's value trained

print(network.params) # i think that are wights
# print(network)

# print(network.activate([-1]))

