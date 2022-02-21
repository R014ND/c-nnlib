using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{   
    class Matrix
    {
        public double[,] values;
        public int rows, cols;

        //constructor
        public Matrix(int rows, int cols, bool isEmpty = false)
        {
            values = new double[rows, cols];
            this.rows = rows;
            this.cols = cols;
            if (!isEmpty)
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        this.values[i, j] = -1;
                    }
                }
            }
        }

        //basic operations

        //fill the matrix with random numbers between 0.0 and 1.0
        public void randomize()
        {
            Random r = new Random();
            //order does not matter but this does it in TOP -> BOTTOM first and then UP->RIGHT order (simply change cycles to do it reversed)
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] = r.NextDouble();
                }
            }
        }
        public void print()
        {
            string line;
            for (int i = 0; i < this.rows; i++)
            {
                line = "";
                for (int j = 0; j < this.cols; j++)
                {
                    line += this.values[i, j] + " ";
                }
                Console.WriteLine(line);
            }
        }
        //set all values in matrix to a given parameter
        public void setValues(double v)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] = v;
                }
            }
        }
        //derive
        public Matrix dSigmoid()
        {
            Matrix output = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    output.values[i, j] = this.values[i, j] * (1 - this.values[i, j]);
                }
            }
            return output;
        }
        //activation function
        public void activate()
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] = 1 / (1 + Math.Pow(Math.E, -this.values[i, j]));
                }
            }
        }
        //transpose
        public Matrix transpose()
        {
            Matrix output = new Matrix(this.cols, this.rows);
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    output.values[j, i] = this.values[i, j];
                }
            }
            return output;
        }

        //scalar operations--------------------------------------------------

        //add parameter s to every value
        public void addScalar(double s)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] += s;
                }
            }
        }
        //subtraction
        public void subtractScalar(double s)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] -= s;
                }
            }
        }
        //multipliy all values by parameter s
        public void multiplyScalar(double s)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] *= s;
                }
            }
        }
        //division, since values are stored in a double array i can do this just for fun
        public void divideScalar(double s)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] /= s;
                }
            }
        }

        //elementwise operations---------------------------------------------
        //operates with elements on the same coordinates (!!!requires 2 matrices with the same dimensions!!!)
        
        //elementwise addition
        public void addMatrix(Matrix a)
        {
            //custom exception for size diff in matrices
            if (this.rows != a.rows || this.cols != a.cols)
            {
                throw new ArgumentException("Matrices should have the same number of rows and the same number of columns.");
            }

            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] += a.values[i, j]; 
                }
            }
        }
        //elementwise multiplication
        public void multiplyMatrix(Matrix a)
        {
            //custom exception for size diff in matrices
            if (this.rows != a.rows || this.cols != a.cols)
            {
                throw new ArgumentException("Matrices should have the same number of rows and the same number of columns.");
            }

            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.values[i, j] *= a.values[i, j];
                }
            }
        }
        //same functions but they return a new matrix as the result of the EW operations
        public Matrix addEW(Matrix a)
        {
            Matrix output = new Matrix(a.rows, a.cols);
            //custom exception for size diff in matrices
            if (this.rows != a.rows || this.cols != a.cols)
            {
                throw new ArgumentException("Matrices should have the same number of rows and the same number of columns.");
            }

            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    output.values[i, j] = this.values[i, j] + a.values[i, j];
                }
            }

            return output;
        }
        public Matrix multiplyEW(Matrix a)
        {
            Matrix output = new Matrix(a.rows, a.cols);
            //custom exception for size diff in matrices
            if (this.rows != a.rows || this.cols != a.cols)
            {
                throw new ArgumentException("Matrices should have the same number of rows and the same number of columns.");
            }

            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    output.values[i, j] = this.values[i, j] * a.values[i, j];
                }
            }

            return output;
        }
        public Matrix subtractMatrix(Matrix a)
        {
            Matrix output = new Matrix(this.rows, this.cols);
            if (this.rows != a.rows || this.cols != a.cols)
            {
                throw new ArgumentException("Matrices should have the same number of rows and the same number of columns.");
            }

            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    output.values[i, j] = this.values[i, j] - a.values[i, j];
                }
            }
            return output;
        }

        //matrix multiplication---------------------------------------------
        //complete mess
        //For matrix multiplication, the number of columns in the first matrix must be equal
        //to the number of rows in the second matrix
        public Matrix dotProduct(Matrix a)
        {
            Matrix output = new Matrix(this.rows, a.cols);
            if (this.cols != a.rows)
            {
                throw new ArgumentException("To calculate the dot product you must have matrices with opposite length in dimensions");
            }

            for (int i = 0; i < output.rows; i++)
            {
                for (int j = 0; j < output.cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < this.cols; k++)
                    {
                        sum += this.values[i, k] * a.values[k, j];
                    }
                    output.values[i, j] = sum;
                }
            }

            return output;
        }
    }
}
