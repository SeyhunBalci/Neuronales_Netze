using Balci_Neuronale_Netze_Woche_1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Balci_Neuronale_Netze_Woche_1
{
    public partial class MainWindow : Window
    {
        int inodes = 3, hnodes = 3, onodes = 3;
        nn3S nn3SO;
        double[] inputs;
        double[] targets;
        double learningRate = 0.1; //learning rate
        
        public MainWindow()
        {
            InitializeComponent();
        }
        private void inputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int inodes);
            Console.WriteLine("input : " + inodes);
        }
        private void hiddenTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int hnodes);
        }
        private void outputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int onodes);
        }
        private void learningRateTextBox_PreviewTextInput(object sender,
       TextCompositionEventArgs e)
        {
            e.Handled = !double.TryParse(e.Text, out double learningRate);
        }
        private void createButton_Click(object sender, RoutedEventArgs e)
        {
            if ((inodes != 0) && (hnodes != 0) && (onodes != 0))
                nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);
        }
        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            inputs = new double[inodes];
            targets = new double[onodes];
            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;
            targets[0] = 0.9;
            targets[1] = 0.9;
            targets[2] = 0.9;
            nn3SO.train(inputs, targets);
            displayResults();

        }

        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            
            TextBox textBox = sender as TextBox;

            
            if (double.TryParse(textBox.Text, out double newLearningRate))
            {
                learningRate = newLearningRate; 
            }
        }

        private void queryButton_Click(object sender, RoutedEventArgs e)
        {
            
            inputs = new double[inodes];
            targets = new double[onodes];
            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;
            targets[0] = 0.9;
            targets[1] = 0.9;
            targets[2] = 0.9;
            nn3SO.train(inputs, targets);
            nn3SO.queryNN(inputs);
            displayResults();

        }

        private void displayResults()
        {
            int weightIHsize = (int)(nn3SO.WIH.Length / inodes);
            int weightHOsize = (int)(nn3SO.WHO.Length / hnodes);
            for (int i = 0; i < inodes; i++)
            {
                // List<double> weightColumn = new List<double>();
                string weightIHColumn = "", weightHOColumn = "";
                for (int j = 0; j < weightIHsize; j++)
                    weightIHColumn += String.Format(" {0:0.##}, ", nn3SO.WIH[i, j]);
                for (int j = 0; j < weightIHsize; j++)
                    weightHOColumn += String.Format(" {0:0.##}, ", nn3SO.WHO[i, j]);
                nodeRow data = new nodeRow
                {
                    inputValue = inputs[i].ToString(),
                    weightsIH = weightIHColumn,
                    inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]),
                    outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]),
                    weightsHO = weightHOColumn,
                    errorHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_errors[i]),
                    inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]),
                    outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]),
                    target = targets[i].ToString(),
                    errorOutput = String.Format(" {0:0.##} ", nn3SO.Output_errors[i]),

                };
                //ComboBoxWeightsIH.ItemsSource = weightColumn;
                networkDataGrid.Items.Add(data);
            }
        }
    }
    public class nodeRow
    {
        public string inputValue { get; set; }
        public string weightsIH { get; set; }
        // public List<double> weightsIH { get; set; }
        // public ComboBox weightsIH { get; set; }
        public string inputHidden { get; set; }
        public string outputHidden { get; set; }
        public string weightsHO { get; set; }
        public string errorHidden { get; set; }
        public string inputOutput { get; set; }
        public string outputLayer { get; set; }
        public string target { get; set; }
        public string errorOutput { get; set; }
    }
}
