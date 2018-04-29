object main{
  def main(args: Array[String]) = {
    var train = Array.ofDim[Double](4,2)
    var target = Array.ofDim[Double](4,1)
    train(0)(0) = 0.0;train(0)(1) = 1.0; target(0)(0) = 1.0
    train(1)(0) = 0.0;train(1)(1) = 0.0; target(1)(0) = 0.0
    train(2)(0) = 1.0;train(2)(1) = 1.0; target(2)(0) = 0.0
    train(3)(0) = 1.0;train(3)(1) = 0.0; target(3)(0) = 1.0

    var nn = new NeuralNetwork(2,4,1)
    nn.init()

    for(i: Int <- 0 until 1000){
      for(t: Int <- 0 until 4){
        nn.insertInput(train(t))
        nn.feedForward()
        nn.train(target(t))
        //nn.printOutput()
      }
    }
    nn.input(0) = 1.0
    nn.input(1) = 1.0
    nn.feedForward()
    nn.printOutput()
  }

}

class NeuralNetwork(inputNodes: Int, hiddenNodes: Int, outputNodes: Int){
  val ins: Int = inputNodes
  val nodes: Int = hiddenNodes
  val outs: Int = outputNodes
  // Neuron values
  var input: Array[Double] = new Array[Double](inputNodes)
  var node: Array[Double] = new Array[Double](nodes)
  var out: Array[Double] = new Array[Double](outs)
  // Weights
  var InNode = Array.ofDim[Double](ins,nodes)
  var NodeOut = Array.ofDim[Double](nodes,outs)
  // Biases
  var NodeBias = new Array[Double](nodes)
  var OutBias = new Array[Double](outs)
  // Settings
  var Learning_Rate = 0.05

  def init() : Unit = {
    val r = new scala.util.Random
    InNode = InNode.map(_.map(_ => r.nextDouble()))
    NodeOut = NodeOut.map(_.map(_ => r.nextDouble()))
    NodeBias = NodeBias.map(_ => r.nextDouble())
    OutBias = OutBias.map(_ => r.nextDouble())
    println("[NN] Values initialized")
  }

  def insertInput(inputs: Array[Double]): Unit = {
    for(i: Int <- 0 until inputNodes) {
      input(i) = inputs(i)
    }
  }

  def feedForward(): Unit ={
    node = node.map(x => 0.0)
    out = out.map(x => 0.0)
    for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
      node(j) += input(i) * InNode(i)(j) + NodeBias(j)
    node = activate(node)
    for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
      out(j) += node(i) * NodeOut(i)(j) + OutBias(j)
    out = activate(out)
  }

  def train(target: Array[Double]): Unit ={
    var IN_Grad = Array.ofDim[Double](ins,nodes)
    var NO_Grad = Array.ofDim[Double](nodes,outs)
    var NB_Grad = new Array[Double](nodes)
    var OB_Grad = new Array[Double](outs)

    var Out_Signal = new Array[Double](outs)
    var Node_Signal = new Array[Double](nodes)

    var derivative: Double = 0.0
    var errorSignal: Double = 0.0

    // Output signals
    for(i: Int <- 0 until outs){
      errorSignal = target(i) - out(i)
      // tanh'(x) = 1 − (tanh(x)E2)
      derivative = 1 - (math.tanh(out(i))*math.tanh(out(i)))
      Out_Signal(i) = errorSignal*derivative
    }
    // Node-Output gradient
    for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
      NO_Grad(i)(j) = Out_Signal(j)*node(i)
    for(i: Int <- 0 until outs) OB_Grad(i) = Out_Signal(i) * 1.0 // duh

    // Hidden node signals
    for(i: Int <- 0 until nodes){
      // tanh'(x) = 1 − (tanh(x)E2)
      derivative = 1 - math.tanh(node(i))*math.tanh(node(i))
      var sum: Double = 0.0
      for(j: Int <- 0 until outs) sum += Out_Signal(j)*NodeOut(i)(j)
      Node_Signal(i) = derivative*sum
    }
    // Input-Hidden gradient
    for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
      IN_Grad(i)(j) = Node_Signal(j) * input(i)
    for(i: Int <- 0 until nodes)
      NB_Grad(i) = Node_Signal(i)*1.0 // duh²

    // Update weights and biases

    // Input - Hidden
      // Weights:
      for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
        InNode(i)(j) += IN_Grad(i)(j) * Learning_Rate
      // Biases:
      for(i: Int <- 0 until nodes)
        NodeBias(i) += NB_Grad(i)*Learning_Rate
    // Hidden - Output
      // Weights:
      for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
        NodeOut(i)(j) += NO_Grad(i)(j) * Learning_Rate
      // Biases:
      for(i: Int <- 0 until outs)
        OutBias(i) += OB_Grad(i) * Learning_Rate
  }

  def printOutput(): Unit ={
    println()
    println("[Outputs]:")
    for(o: Double <- out)
      print(" "+o)
  }

  def activate(arr: Array[Double]) ={
    //arr.map(x => math.tanh(x)) // Hyper-Tan, for logic gates (like XOR) it's preferable to use Step Function
    arr.map{x => if(x > 0.5) 1.0 else 0.0}
  }

  def softmax(arr: Array[Double]): Unit = {
    var sum: Double = 0.0
    @scala.annotation.tailrec
    def go(a: Array[Double]): Unit ={
      sum += a.head
      if(a.tail.isEmpty == false) go(a.tail)
    }
    var res = arr.clone()
    res.map(x => math.exp(x)/sum)
  }

}