package neuralNetWork;

import classify.Classify;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.classifier.mlp.NeuralNetworkFunctions;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import static tool.Tool.labelIndex;
import static tool.Tool.paths;

/**
 * Created by sghipr on 5/11/16.
 * 通过Error BP算法来训练出一个神经网络模型.
 * 这里设置隐藏层的激活函数为sigmoid函数.
 * sigmoid函数有一个性质:
 * h'(x) = h(x)*(1 - h(x))
 *
 * Regularizer = alhpa1/2 * sum(firstLayerWeights^2) + alpha2/2 * sum(secondLayerWeights^2)
 * 主要是为了确保当输入向量进行某种线性平移时，不会造成模型的改变.因此没有采用L2范数.
 */
public class TrainNeuralNetWork extends Classify{

    /**
     * 这里默认的训练数据的格式为:classLabel,featureVector;
     * trainPath可能是一个目录也可能是一个文件.
     */
    //private Path trainPath;

    /**
     * 最大地迭代次数.
     */
    private int maxIters = 36;

    private Configuration conf;

    private NeuralNetWorkModel netWorkModel;

    private HashMap<String,Integer> labelMap;

    private double regularWeight = 0.001;

    /**
     * 设置的隐藏层中节点的个数.
     * 默认为六个隐藏层地结点.
     */
    private int M = 6;

    /**
     *正则化因子.
     */
    private double lambda1 = 1e-6;

    private double lambda2 = 1e-6;

    /**
     * 学习率.
     */
    private double learningRate = 0.01;

    public void setM(int M){
        this.M = M;
    }

    public void setMaxIters(int iters){
        this.maxIters = iters;
    }

    public void setLambda(double lambda1, double lambda2){
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }

    public void setLearningRate(double learningRate){
        this.learningRate = learningRate;
    }

    public void setRegularWeight(double weight){
        this.regularWeight = weight;
    }

    public TrainNeuralNetWork(){
        conf = new Configuration();
        netWorkModel = new NeuralNetWorkModel();
    }

    @Override
    public void buildClassify(Path files) {

        Path labelPath = null;
        List<Path> trainPaths = new ArrayList<>();
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
        boolean initilizeFlag = true;

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
                        if (initilizeFlag) {
                            netWorkModel.initializeModel(features.get().size(), M, labelMap.size(), labelMap);
                            initilizeFlag = false;
                        }
                        boolean success = trainByGradient(label.toString(), features.get());
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
    /**
     * 随机梯度下降.
     * 通过error BP反向传播算法来优化其权值.
     * 注意,隐藏层的激活函数为sigmoid函数.
     * d(En/wji) = d(En/aj) * d(aj/wji) = d(En/aj) * zi;
     *
     * @param label
     * @param features
     * @return sumError 每次经过神经网络分类时的总的误差.
     */
    public boolean trainByGradient(String label, Vector features){

        //forward information
        Vector[] outputs = new DenseVector[3];
        outputs[0] = features;
        for(int layer = 0; layer < 2; ++layer){
            features = netWorkModel.forward(layer, features);
            outputs[layer + 1] = features;
        }

        //yk same as zk
        Vector yk = outputs[2];
        Vector target = new DenseVector(labelMap.size());
        target.set(labelMap.get(label), 1.0);
        Vector derivateAk = new DenseVector(target.size());
        for(int i = 0; i < derivateAk.size(); ++i){
            double derivateCostFunction = NeuralNetworkFunctions.derivativeCrossEntropy.apply(target.get(i),yk.get(i));
            derivateCostFunction = derivateCostFunction * Functions.SIGMOIDGRADIENT.apply(yk.get(i));
            derivateAk.set(i,derivateCostFunction);
        }
        //来自mahout
//        for(int i = 0; i < derivateAk.size(); i++)
//            derivateAk.set(i, derivateAk.get(i) + regularWeight * netWorkModel.getFirstLayerWeights().viewRow(i).zSum());

        //backPropagate
        Matrix[] weightUpdateMatrixs = new Matrix[2];
        Matrix[] modelWeights = new Matrix[2];
        modelWeights[0] = netWorkModel.getFirstLayerWeights();
        modelWeights[1] = netWorkModel.getSecondLayerWeights();
        double[] lambdas = {lambda1,lambda2};

        weightUpdateMatrixs[0] = new DenseMatrix(netWorkModel.getFirstLayerWeights().rowSize(),netWorkModel.getFirstLayerWeights().columnSize());
        weightUpdateMatrixs[1] = new DenseMatrix(netWorkModel.getSecondLayerWeights().rowSize(),netWorkModel.getSecondLayerWeights().columnSize());
        Vector delta = derivateAk;
        for(int layer = 1; layer >=0 ; --layer){
            delta = backPropagate(delta,outputs[layer],weightUpdateMatrixs[layer],modelWeights[layer]);
        }

        //update
        updateWeights(modelWeights,weightUpdateMatrixs,lambdas);
        return labelMap.get(label) == yk.maxValueIndex();
    }

    public Vector backPropagate(Vector nextLayerDelta, Vector curLayerOut, Matrix weightUpdateMatrix, Matrix modelWeight){

        Vector delta = (modelWeight.transpose().times(nextLayerDelta)).viewPart(1, curLayerOut.size());
        for(int i = 0; i < delta.size(); ++i){
            double derivateFunction = Functions.SIGMOIDGRADIENT.apply(curLayerOut.get(i));
            delta.set(i, derivateFunction * delta.get(i));
        }

        Vector addBais = new DenseVector(curLayerOut.size() + 1);
        addBais.set(0,1.0);
        addBais.viewPart(1,curLayerOut.size()).assign(curLayerOut);

        for(int row = 0; row < weightUpdateMatrix.rowSize(); ++row)
            for(int column = 0; column < weightUpdateMatrix.columnSize(); ++column)
                weightUpdateMatrix.set(row,column, nextLayerDelta.get(row) * addBais.get(column));
        return delta;
    }

    public void updateWeights(Matrix[] modelWeights,Matrix[] weightUpdateMatrixs,double[] lambdas){

        for(int i = 0; i < modelWeights.length; ++i)
            updateWeights(modelWeights[i],weightUpdateMatrixs[i],lambdas[i]);
    }

    public void updateWeights(Matrix modelWeights, Matrix updateWeights, double lambda){

        for(int row = 0; row < modelWeights.rowSize(); ++row)
            for(int column = 0; column < modelWeights.columnSize(); ++column)
                modelWeights.set(row,column, modelWeights.get(row,column) - learningRate * (updateWeights.get(row,column) + lambda * modelWeights.get(row,column)));
    }
}
