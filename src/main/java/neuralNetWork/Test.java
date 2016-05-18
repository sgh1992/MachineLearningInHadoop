package neuralNetWork;

import org.apache.hadoop.fs.Path;

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
        trainNeuralNetWork.setMaxIters(20000);
        trainNeuralNetWork.setLambda(1e-4,1e-4);
        trainNeuralNetWork.setLearningRate(0.001);
        //trainNeuralNetWork.setRegularWeight(0.001);
        trainNeuralNetWork.buildClassify(trains);
    }

    public static void trainMahoutNeuralNetWork(){
        Path trains = new Path("/home/sghipr/careerPredict/trains");
        MahoutNeuralNetWork mahoutNeuralNetWork = new MahoutNeuralNetWork();
        mahoutNeuralNetWork.addLayer(102,false,"");
        mahoutNeuralNetWork.addLayer(32,false,"Sigmoid");
        mahoutNeuralNetWork.addLayer(5,true,"Sigmoid");
        //mahoutNeuralNetWork.setLearningRate(0.08);
        mahoutNeuralNetWork.setRegularWeight(0);
        mahoutNeuralNetWork.setMoment(0);
        mahoutNeuralNetWork.setCostFunction("Cross_Entropy");
        mahoutNeuralNetWork.buildClassify(trains);
    }
}
