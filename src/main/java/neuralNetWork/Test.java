package neuralNetWork;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;

/**
 * Created by sghipr on 5/11/16.
 */
public class Test {

    public static void main(String[] args){

        trainSelfNeuralNetWork();
        //trainMahoutNeuralNetWork();
    }

    public static void trainSelfNeuralNetWork(){
        Path trains = new Path("/home/sghipr/careerPredict/trains");
        TrainNeuralNetWork trainNeuralNetWork = new TrainNeuralNetWork();
        trainNeuralNetWork.setM(16);
        trainNeuralNetWork.setMaxIters(500);
        trainNeuralNetWork.setLambda(1e-3,1e-3);
        trainNeuralNetWork.setLearningRate(0.01);
        trainNeuralNetWork.buildClassify(trains);
    }

    public static void trainMahoutNeuralNetWork(){
        Path trains = new Path("/home/sghipr/careerPredict/trains");
        MahoutNeuralNetWork mahoutNeuralNetWork = new MahoutNeuralNetWork();
        mahoutNeuralNetWork.addLayer(102,false,"");
        mahoutNeuralNetWork.addLayer(32,false,"Sigmoid");
        mahoutNeuralNetWork.addLayer(5,true,"Sigmoid");
        //mahoutNeuralNetWork.setLearningRate(0.08);
        //mahoutNeuralNetWork.setRegularWeight(0.001);
        mahoutNeuralNetWork.setCostFunction("Cross_Entropy");
        mahoutNeuralNetWork.buildClassify(trains);
    }
}
