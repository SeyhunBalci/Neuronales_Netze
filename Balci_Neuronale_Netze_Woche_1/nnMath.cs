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


        public double[,] transpose(double[,] inputMat)
        {
            double[,] results = new double[inputMat.GetLength(1), inputMat.GetLength(0)];

            for (int i = 0; i < inputMat.GetLength(0); i++) // Zeilen 
            {
                for (int j = 0; j < inputMat.GetLength(1); j++) // Spalten
                {
                    results[j, i] = inputMat[i, j];
                }
            }
            return results;
        }

        

        public double[] sigmoidDerivative(double[] outputs)
        {
            double[] results = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                results[i] = outputs[i] * (1 - outputs[i]); 
            }
            return results;
        }


        
        public double[] vectorMult(double[] vec1, double[] vec2)
        {
            
            double[] result = new double[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
            {
                result[i] = vec1[i] * vec2[i];
            }
            return result;
        }




        public double[,] fullMatrixMult(double[] vec1, double[] vec2)
        {
            
            double[,] result = new double[vec1.Length, vec2.Length];

            for (int i = 0; i < vec1.Length; i++) //Spaltenvektor
            {
                for (int j = 0; j < vec2.Length; j++) //Zeilenvektor
                {
                    result[i, j] = vec1[i] * vec2[j]; 
                }
            }
            return result;
        }


        // Sum two matrices element-wise
        public double[,] matrixSum(double[,] mat1, double[,] mat2)
        {
            
            // Ergebnis-Matrix initialisieren
            double[,] result = new double[mat1.GetLength(0), mat1.GetLength(1)];

            // Elementweise Addition
            for (int i = 0; i < mat1.GetLength(0); i++) //Zeilen
            {
                for (int j = 0; j < mat1.GetLength(1); j++) //Spalten
                {
                    result[i, j] = mat1[i, j] + mat2[i, j];
                }
            }
            return result;
        }


        // Multiply element-wise a matrix by a scaling factor
        public double[,] matrixScale(double[,] inputMat, double scale)
        {
            // Ergebnis-Matrix initialisieren
            double[,] result = new double[inputMat.GetLength(0), inputMat.GetLength(1)];

            // Elementweise Skalierung
            for (int i = 0; i < inputMat.GetLength(0); i++) //Zeilen
            {
                for (int j = 0; j < inputMat.GetLength(1); j++) //Spalten
                {
                    result[i, j] = inputMat[i, j] * scale; 
                }
            }
            return result;
        }







    }
}
