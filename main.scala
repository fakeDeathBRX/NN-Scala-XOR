object main{
    def main(args: Array[String]) = {
        var nn = new NeuralNetwork(2,30,1)
        val epoch = 5000 // 5000 is good, but sometimes it requires a little more
        val samples = epoch*0.01
        var error = 0.0
        nn.init()
        for(i: Int <- 1 until epoch*5000){
            val r = new scala.util.Random
            var angle: Double = r.nextDouble()
            var train = Array.ofDim[Double](2)
            var target = Array.ofDim[Double](1)
            train(0) = math.sin(angle)
            train(1) = math.cos(angle)
            //train(2) = math.tan(angle)
            target(0) = angle
            nn.insertInput(train)
            error += nn.train(target)
            if(i%samples == 0){
                error /= samples
                print(s"LR: ${nn.Learning_Rate} Error: $error\r")
                error = 0.0
            }
            if(i%(epoch*1000) == 0)
                nn.Learning_Rate /= 2
        }
        println()
        val r = new scala.util.Random
        var angle: Double = r.nextDouble()
        var test = Array.ofDim[Double](2)
        var tar: Double = angle
        test(0) = math.sin(angle)
        test(1) = math.cos(angle)
        //test(2) = math.tan(angle)
        nn.insertInput(test)
        nn.feedForward()
        nn.printOutput()
        println(s"Target output: $tar")
    }

}

class NeuralNetwork(inputNodes: Int, hiddenNodes: Int, outputNodes: Int){
    // Class related
    val ins: Int = inputNodes
    val nodes: Int = hiddenNodes
    val outs: Int = outputNodes

    // Neuron values
    var input: Array[Double] = new Array[Double](ins)
    var node: Array[Double] = new Array[Double](nodes)
    var out: Array[Double] = new Array[Double](outs)

    // Weights
    var InNode = Array.ofDim[Double](ins,nodes)
    var NodeOut = Array.ofDim[Double](nodes,outs)

    // Biases
    var NodeBias = new Array[Double](nodes)
    var OutBias = new Array[Double](outs)

    // Settings
    var Learning_Rate = 0.001

    def init(): Unit = {
        val r = new scala.util.Random
        InNode = InNode.map(_.map(_ => r.nextDouble()))
        NodeOut = NodeOut.map(_.map(_ => r.nextDouble()))
        NodeBias = NodeBias.map(_ => r.nextDouble())
        OutBias = OutBias.map(_ => r.nextDouble())
        println("[NN] Values initialized")
    }

    def insertInput(inputs: Array[Double]): Unit = {
        for(i: Int <- 0 until ins)
            input(i) = inputs(i)
    }

    def feedForward(): Unit ={
        node = node.map(x => 0.0)
        out = out.map(x => 0.0)

        for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
            node(j) += input(i) * InNode(i)(j) + NodeBias(j)
        for(i: Int <- 0 until nodes)
            node(i) = activate(node(i))

        for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
            out(j) += node(i) * NodeOut(i)(j) + OutBias(j)
        for(i: Int <- 0 until outs)
            out(i) = activate(out(i))
    }

    def train(target: Array[Double]): Double ={
        feedForward()
        // Gradients
        var IN_Grad = Array.ofDim[Double](ins,nodes) // Input-Hidden
        var NO_Grad = Array.ofDim[Double](nodes,outs) // Hidden-Output
        var NB_Grad = new Array[Double](nodes) // Hidden Bias
        var OB_Grad = new Array[Double](outs) // Output Bias

        var Out_Signal = new Array[Double](outs) // Output signal
        var Node_Signal = new Array[Double](nodes) // Hidden signal

        var derivative: Double = 0.0
        var errorSignal: Double = 0.0

        var error: Double = 0.0

        // Output signals
        for(i: Int <- 0 until outs){
            errorSignal = target(i) - out(i)
            derivative = 1 - (math.tanh(out(i))*math.tanh(out(i)))
            Out_Signal(i) = errorSignal*derivative
            error += errorSignal
        }

        // Node-Output gradient
        for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
            NO_Grad(i)(j) = Out_Signal(j)*node(i)
        for(i: Int <- 0 until outs)
            OB_Grad(i) = Out_Signal(i) * 1.0

        // Hidden node signals
        for(i: Int <- 0 until nodes){
            var sum: Double = 0.0
            derivative = 1 - math.tanh(node(i))*math.tanh(node(i))
            for(j: Int <- 0 until outs)
                sum += Out_Signal(j)*NodeOut(i)(j)
            Node_Signal(i) = derivative*sum
        }

        // Input-Hidden gradient
        for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
            IN_Grad(i)(j) = Node_Signal(j) * input(i)
        for(i: Int <- 0 until nodes)
            NB_Grad(i) = Node_Signal(i)*1.0

        // Update weights and biases

        // Input - Hidden
        // Weights:
        for(i: Int <- 0 until ins; j: Int <- 0 until nodes)
            InNode(i)(j) += IN_Grad(i)(j) * Learning_Rate
        // Biases:
        for(i: Int <- 0 until nodes)
            NodeBias(i) += NB_Grad(i) * Learning_Rate

        // Hidden - Output
        // Weights:
        for(i: Int <- 0 until nodes; j: Int <- 0 until outs)
            NodeOut(i)(j) += NO_Grad(i)(j) * Learning_Rate
        // Biases:
        for(i: Int <- 0 until outs)
            OutBias(i) += OB_Grad(i) * Learning_Rate
        return math.abs(error)
    }

    def printOutput(): Unit ={
        println("[Outputs]:")
        for(o: Double <- out)
            print(s"$o")
        println()
    }

    def activate(x: Double): Double ={
        return math.tanh(x)
    }

    def sigmoid(x: Double): Double ={
        return 1/(1+math.exp(-x))
    }
}
