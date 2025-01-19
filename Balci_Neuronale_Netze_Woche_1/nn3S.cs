using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;


namespace Balci_Neuronale_Netze_Woche_1
{
    class nn3S
    {
        double[,] wih, who;
        int inodes, hnodes, onodes;
        double learningRate;
        double[] hidden_inputs;
        double[] hidden_outputs;
        double[] final_inputs;
        double[] final_outputs;

        double[] hidden_errors;
        double[] output_errors;

        public double[] Hidden_inputs { get { return hidden_inputs; } }
        public double[] Hidden_outputs { get { return hidden_outputs; } }
        public double[] Final_inputs { get { return final_inputs; } }
        public double[] Final_outputs { get { return final_outputs; } }


        public double[,] WIH { get { return wih; } }
        public double[,] WHO { get { return who; } }

        public double[] Hidden_errors {  get { return hidden_errors; } }
        public double[] Output_errors {  get { return output_errors; } }


        public nn3S(int inodes, int hnodes, int onodes, double learningRate)
        {
            this.inodes = inodes;
            this.hnodes = hnodes;
            this.onodes = onodes;
            this.learningRate = learningRate;

            createWeightMatrizes();

            
        }


        public void train(double[] inputs, double[] targets)
        {
            // In dieser Funktion laufen wir die Schritte der beschriebenen Zweiten Vorlesung NN auf der letzte Seite einmal ab! 
            nnMath nnMathO = new nnMath();

            //1 
            //Als erstes brauchen wir die Ausgabedaten.
            queryNN(inputs);

            //----------------------------------------------------------------
            //2 
            //Danach müssen wir den Error brechnen! e1 = (tk - Ok)
            output_errors = new double[onodes];
            for (int i = 0; i < onodes; i++)
            {
                output_errors[i] = targets[i] - final_outputs[i];
            }

            //----------------------------------------------------------------

            //3
            //Jetzt haben wir letzten Error an unserem NN jetzt müssen wir die versteckten Schicht Error finden. 
            //Seite 9 aus der Vorlesung 2 NN 
            //e_hidden = wT * e_out 
            double[,] who_T = nnMathO.transpose(who);
            hidden_errors = nnMathO.matrixMult(who_T, hnodes, output_errors);

            //----------------------------------------------------------------

            // 4.
            // Aktualisierung der Gewichte zwischen der versteckten Schicht und der Ausgabeschicht

            // Berechnung der Ableitung der Sigmoidfunktion
            double[] sigmoid_derivative = nnMathO.sigmoidDerivative(final_outputs);

            // Berechnung des Gradienten der Ausgabeschicht
            double[] output_gradient = nnMathO.vectorMult(output_errors, sigmoid_derivative);

            // Berechnung der Gewichtsanpassung für WHO 
            double[,] delta_who = nnMathO.fullMatrixMult(output_gradient, hidden_outputs);

            // Aktualisierung der Gewichte
            who = nnMathO.matrixSum(who, nnMathO.matrixScale(delta_who, learningRate));

            // 5. 
            //Aktualisierung der Gewichte zwischen der Eingabeschicht und der versteckten Schicht 
            // ist das gelice wie oben im Grunde 
            double[] sigmoid_hidden_derivative = nnMathO.sigmoidDerivative(hidden_outputs);

             //Berechnung des Gradienten der versteckten Schicht
            double[] hidden_gradient = nnMathO.vectorMult(hidden_errors, sigmoid_hidden_derivative);

            //Berechnung der Gewichtsanpassung
            double[,] delta_wih = nnMathO.fullMatrixMult(hidden_gradient, inputs);

            //Aktualisierung der Gewichte
            wih = nnMathO.matrixSum(wih, nnMathO.matrixScale(delta_wih, learningRate));

        }


        private void createWeightMatrizes()
        {
            wih = new double[inodes, hnodes];
            who = new double[hnodes, onodes];

            //Diese habe ich getauscht sonst komme ich nicht auf den Wert!
            //In der vorgegebenen datei ist Splate mit Zeile vertauscht! 
            wih[0, 0] = 0.9;
            wih[0, 1] = 0.3;
            wih[0, 2] = 0.4;
            wih[1, 0] = 0.2;
            wih[1, 1] = 0.8;
            wih[1, 2] = 0.2;
            wih[2, 0] = 0.1;
            wih[2, 1] = 0.5;
            wih[2, 2] = 0.6;
            

            who[0, 0] = 0.3;
            who[0, 1] = 0.7;
            who[0, 2] = 0.5;
            who[1, 0] = 0.6;
            who[1, 1] = 0.5;
            who[1, 2] = 0.2;
            who[2, 0] = 0.8;
            who[2, 1] = 0.1;
            who[2, 2] = 0.9;

            
        }

        public void queryNN(double[] inputs)
        {
            nnMath nnMathO = new nnMath();
            
            hidden_inputs = new double[hnodes];
            hidden_inputs = nnMathO.matrixMult(wih, inodes, inputs);

            hidden_outputs = new double[hnodes];
            hidden_outputs = nnMathO.activationFunction(hidden_inputs);

            final_inputs = new double[onodes];
            final_inputs = nnMathO.matrixMult(who, hnodes, hidden_outputs);

            final_outputs = new double[hnodes];
            final_outputs = nnMathO.activationFunction(final_inputs);
        }
    }
}


