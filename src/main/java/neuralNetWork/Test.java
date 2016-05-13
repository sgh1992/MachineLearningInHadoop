package neuralNetWork;

import org.apache.hadoop.fs.Path;

/**
 * Created by sghipr on 5/11/16.
 */
public class Test {

    public static void main(String[] args){

        Path trains = new Path("/home/sghipr/careerPredict/trains");
        TrainNeuralNetWork trainNeuralNetWork = new TrainNeuralNetWork();
        trainNeuralNetWork.setM(36);
        trainNeuralNetWork.setMaxIters(500);
        trainNeuralNetWork.setLambda(1e-4,1e-4);
        //trainNeuralNetWork.setLearningRate(0.1);
        trainNeuralNetWork.buildClassify(trains);

    }

}
