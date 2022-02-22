using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    class NeuralNetwork
    {
        public string name;
        //weight notation : weight(mn) => weight between layer m & layer n
        //weight matrix of a connection network between layers has
        //as many columns as the layer to the right and as many rows as the layer to the left 
        public int input_nodes;
        public int hidden_nodes;
        public int output_nodes;

        public Matrix weights_ih;
        public Matrix weights_ho;
        public Matrix bias_hidden;
        public Matrix bias_output;

        public double learningRate;

        public NeuralNetwork(string name, int input, int hidden, int output , double learningRate = 0.1)
        {
            this.name = name;
            this.input_nodes = input;
            this.hidden_nodes = hidden;
            this.output_nodes = output;
            this.learningRate = learningRate;

            this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
            this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);

            //set weights to completely random numbers between 0.0 & 1.0
            this.weights_ih.randomize();
            this.weights_ho.randomize();

            //bias nodes
            this.bias_hidden = new Matrix(this.hidden_nodes, 1);
            this.bias_output = new Matrix(this.output_nodes, 1);
            this.bias_hidden.randomize();
            this.bias_output.randomize();
        }
        //feedforward algorithm...
        //core of the neural network
        public Matrix feedForward(double[] input_array)
        {

            //XOR output must be close to 1 or 0
            //in case of using this for something else you may change it to some other totally unreal value
            //making a matrix from the input array
            Matrix inputs = this.fromArray(input_array);

            //calculating the values of the left side connections
            Matrix hidden = this.weights_ih.dotProduct(inputs);
            //adding bias
            hidden.addMatrix(this.bias_hidden);
            //map all values between 0.0 and 1.0 (sigmoid is logarithmic so actually it never reaches 0 nor 1)
            hidden.activate();


            //calculating values for the right side connections
            Matrix output = this.weights_ho.dotProduct(hidden);
            output.addMatrix(this.bias_output);
            output.activate();

            //in this case we have only one 
            Matrix guess = output;

            return guess;
        }

        //train the network by adjusting weights based on the calculated error, aka.: supervised learning
        public void train(double[] input_array, double[] target_array)
        {
            //XOR output must be close to 1 or 0
            //in case of using this for something else you may change it to some other totally unreal value
            //making a matrix from the input array
            Matrix inputs  = fromArray(input_array);
            Matrix targets = fromArray(target_array);

            //calculating the values of the left side connections
            Matrix hidden = this.weights_ih.dotProduct(inputs);
            //adding bias
            hidden.addMatrix(this.bias_hidden);
            //map all values between 0.0 and 1.0 (sigmoid is logarithmic so actually it never reaches 0 nor 1)
            hidden.activate();


            //calculating values for the right side connections
            Matrix outputs = this.weights_ho.dotProduct(hidden);
            outputs.addMatrix(this.bias_output);
            outputs.activate();

            //train----------------------------------------------------------------------------------------
            
            //calculating the error based on the answer to a given input
            //diff between the targets and the outputs
            Matrix output_errors = targets.subtractMatrix(outputs);

            //calculate gradient
            //derive outputs
            Matrix gradients = outputs.dSigmoid();
            gradients.multiplyMatrix(output_errors);

            //scaling deltas by lr
            gradients.multiplyScalar(this.learningRate);

            //adjusting bias by its deltas which is simply gradients
            this.bias_output.addMatrix(gradients);

            Matrix hidden_T = hidden.transpose();
            Matrix weights_ho_deltas = gradients.dotProduct(hidden_T);

            this.weights_ho.addMatrix(weights_ho_deltas);

            //----------------------------



            //hidden errors
            Matrix weights_ho_t = this.weights_ho.transpose();
            Matrix hidden_errors = weights_ho_t.dotProduct(output_errors);

            Matrix hidden_gradients = hidden.dSigmoid();

            hidden_gradients.multiplyMatrix(hidden_errors);
            hidden_gradients.multiplyScalar(this.learningRate);

            Matrix inputs_T = inputs.transpose();

            Matrix weights_ih_deltas = hidden_gradients.dotProduct(inputs_T);

            //adjusting bias by its deltas
            this.bias_hidden.addMatrix(hidden_gradients);

            this.weights_ih.addMatrix(weights_ih_deltas);

        }

        //matrix from array--------------------------------------------------
        public Matrix fromArray(double[] array)
        {
            Matrix output = new Matrix(array.Length, 1);
            for (int i = 0; i < array.Length; i++)
            {
                output.values[i, 0] = array[i];
            }
            return output;
        }
        //array from matrix
        public double[] toArray(Matrix m)
        {
            double[] output = new double[m.values.GetLength(0)];
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = m.values[i, 0];
            }
            return output;
        }
    }
}
