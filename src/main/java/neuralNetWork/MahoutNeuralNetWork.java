package neuralNetWork;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import static tool.Tool.labelIndex;
import static tool.Tool.paths;

/**
 * Created by sghipr on 15/05/16.
 * 使用Mahout的NeuralNetWork来进行比较.
 */
public class MahoutNeuralNetWork {

    private Configuration conf;

    private MultilayerPerceptron mlp;

    private int maxIters = 500;

    public void addLayer(int size, boolean isFinall, String activation){
        mlp.addLayer(size,isFinall,activation);
    }

    public MahoutNeuralNetWork(){
        mlp = new MultilayerPerceptron();
        conf = new Configuration();
    }

    public void setMoment(double moment){
        mlp.setMomentumWeight(moment);
    }

    public void setLearningRate(double learningRate){
        mlp.setLearningRate(learningRate);
    }

    public void setRegularWeight(double regularWeight){
        mlp.setRegularizationWeight(regularWeight);
    }

    public void setMaxiters(int iters){
        this.maxIters = iters;
    }

    public void setCostFunction(String costFunction){
        mlp.setCostFunction(costFunction);
    }

    public void buildClassify(Path files){

        Path labelPath = null;
        List<Path> trainPaths = new ArrayList<>();
        HashMap<String,Integer> labelMap = null;
        try {
            labelPath = NeuralNetWorkJob.runLabelIndexJob(conf,files);
            labelMap = labelIndex(labelPath,conf);
            paths(files,trainPaths,conf);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        SequenceFile.Reader reader = null;
        int iters = 0;
        int totalCount = 0;
        while (iters++ < maxIters){
            int count = 0;
            totalCount = 0;
            for(Path path : trainPaths) {
                try {
                    reader = new SequenceFile.Reader(FileSystem.get(conf), path, conf);
                    Text label = (Text) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
                    VectorWritable features = (VectorWritable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
                    while (reader.next(label, features)) {
                        Vector inst = new DenseVector(features.get().size() + labelMap.size());
                        inst.viewPart(0,features.get().size()).assign(features.get());
                        Vector classVector = new DenseVector(labelMap.size());
                        classVector.set(labelMap.get(label.toString()),1);
                        inst.viewPart(features.get().size(),labelMap.size()).assign(classVector);
                        mlp.trainOnline(inst);
                        boolean success = mlp.getOutput(features.get()).maxValueIndex() == labelMap.get(label.toString());
                        if(success)
                            count++;
                        totalCount++;
                    }
                    reader.close();
                } catch (EOFException e) {
                    //e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            System.err.printf("Iters:%d,precise:%f\n", iters, count/(totalCount + 1.0));
        }
    }
}
