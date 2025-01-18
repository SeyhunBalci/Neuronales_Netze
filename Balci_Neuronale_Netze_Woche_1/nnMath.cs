using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Balci_Neuronale_Netze_Woche_1
{
    internal class nnMath
    {

        public double[] matrixMult(double[,] matrix, int rows, double[] vector)
        {
            
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < vector.Length; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            }

            return result;
        }


        public double[] activationFunction(double[] inputs)
        {
            
            double[] results = new double[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                results[i] = 1 / (1 + Math.Exp(-inputs[i]));
            }
            return results;
        }

    }
}
