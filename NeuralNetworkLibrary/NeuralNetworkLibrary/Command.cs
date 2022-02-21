using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    struct training_data
    {
        public double[] inputs;
        public double[] target;
    }
    class Command
    {

        public List<NeuralNetwork> nns = new List<NeuralNetwork>();
        string command;
        string[] attributes;
        string prefix = "!";
        
        public Command(string prefix, string command, string[] att)
        {
            #region filling data arr
            training_data[] training_Datas = new training_data[16];
            //AND
            training_Datas[0].inputs = new double[] { 0, 0 };
            training_Datas[0].target = new double[] { 0 };

            training_Datas[1].inputs = new double[] { 1, 0 };
            training_Datas[1].target = new double[] { 0 };

            training_Datas[2].inputs = new double[] { 1, 1 };
            training_Datas[2].target = new double[] { 1 };

            training_Datas[3].inputs = new double[] { 0, 1 };
            training_Datas[3].target = new double[] { 0 };

            //NAND
            training_Datas[4].inputs = new double[] { 0, 0 };
            training_Datas[4].target = new double[] { 1 };

            training_Datas[5].inputs = new double[] { 1, 0 };
            training_Datas[5].target = new double[] { 1 };

            training_Datas[6].inputs = new double[] { 1, 1 };
            training_Datas[6].target = new double[] { 0 };

            training_Datas[7].inputs = new double[] { 0, 1 };
            training_Datas[7].target = new double[] { 1 };

            //OR
            training_Datas[8].inputs = new double[] { 0, 0 };
            training_Datas[8].target = new double[] { 0 };

            training_Datas[9].inputs = new double[] { 1, 0 };
            training_Datas[9].target = new double[] { 1 };

            training_Datas[10].inputs = new double[] { 1, 1 };
            training_Datas[10].target = new double[] { 1 };

            training_Datas[11].inputs = new double[] { 0, 1 };
            training_Datas[11].target = new double[] { 1 };

            //XOR
            training_Datas[12].inputs = new double[] { 0, 0 };
            training_Datas[12].target = new double[] { 0 };

            training_Datas[13].inputs = new double[] { 1, 0 };
            training_Datas[13].target = new double[] { 1 };

            training_Datas[14].inputs = new double[] { 1, 1 };
            training_Datas[14].target = new double[] { 0 };

            training_Datas[15].inputs = new double[] { 0, 1 };
            training_Datas[15].target = new double[] { 1 };
            #endregion

            if (this.prefix != prefix)
            {
                Console.WriteLine("Command not found.");
            }
            switch (command)
            {
                case "newNN":
                    bool exists = false;
                    for (int i = 0; i < nns.Count; i++)
                    {
                        if (nns[i].name == att[0])
                        {
                            Console.WriteLine("Neural network under the name " + att[0] + " already exists");
                            exists = true;
                        }
                    }
                    if (!exists)
                    {
                        NeuralNetwork nn = new NeuralNetwork(att[0], int.Parse(att[1]), int.Parse(att[2]), int.Parse(att[3]));
                        this.nns.Add(nn);
                        Console.WriteLine("New neural network " + nn.name + " has been created");
                    }
                    break;
                case "list":
                    for (int i = 0; i < this.nns.Count; i++)
                    {
                        Console.WriteLine(nns[i].name + " (" + nns[i].input_nodes + ", " + nns[i].hidden_nodes + ", " + nns[i].output_nodes + ")");
                    }
                    break;
                case "trainNN":
                    int index = -1;
                    for (int i = 0; i < nns.Count; i++)
                    {
                        if (nns[i].name == att[0])
                        {
                            index = i;
                        }
                    }
                    switch (att[1])
                    {
                        case "AND":
                            for (int i = 0; i < int.Parse(att[2]); i++)
                            {
                                Random r = new Random();
                                int rand = r.Next(0, 4);
                                nns[index].train(training_Datas[rand].inputs, training_Datas[rand].target);
                            }
                            break;
                        case "NAND":
                            for (int i = 0; i < int.Parse(att[2]); i++)
                            {
                                Random r = new Random();
                                int rand = r.Next(4, 8);
                                nns[index].train(training_Datas[rand].inputs, training_Datas[rand].target);
                            }
                            break;
                        case "OR":
                            for (int i = 0; i < int.Parse(att[2]); i++)
                            {
                                Random r = new Random();
                                int rand = r.Next(8, 12);
                                nns[index].train(training_Datas[rand].inputs, training_Datas[rand].target);
                            }
                            break;
                        case "XOR":
                            for (int i = 0; i < int.Parse(att[2]); i++)
                            {
                                Random r = new Random();
                                int rand = r.Next(12, 16);
                                nns[index].train(training_Datas[rand].inputs, training_Datas[rand].target);
                            }
                            break;
                    }
                    Console.WriteLine("Training done on " + att[0] + " network");
                    break;
                default:
                    Console.WriteLine("Command not found");
                    break;
            }
        }
    }
}
