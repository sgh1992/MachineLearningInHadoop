package neuralNetWork;

import classify.Classify;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

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
            labelMap = labelIndex(labelPath);
            paths(files,trainPaths);
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
        double sumErrors = 10000;
        Random random = new Random(17);
        while (iters++ < maxIters){
            sumErrors = 0.0;
            int count = 0;
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
//                        if(random.nextBoolean()){
//                        }
                        boolean success = trainByErrorBP(label.toString(), features.get());
                        if(success)
                            count++;
                    }
                    reader.close();
                } catch (EOFException e) {
                    //e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            System.err.printf("Iters:%d,sumErrors:%d\n", iters, count);
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
    public boolean trainByErrorBP(String label, Vector features){

        Vector Aj = netWorkModel.hiddenUnitInput(features);
        Vector Zj = netWorkModel.hiddenUnitActivate(features);

        //something named ak,ignored
        //yk same as zk
        Vector yk = netWorkModel.predict(features);

        Vector target = new DenseVector(labelMap.size());
        target.set(labelMap.get(label), 1.0);
        Vector derivateAk = yk.minus(target);//yk - tk
        Vector derivateAj = getDerivateAJ(Zj, derivateAk);

        //初始化连接输入层与隐藏层的权重的偏导数值.
        Matrix derivateWeightConnectInputAndHiddenMatrix = new DenseMatrix(netWorkModel.getM(), netWorkModel.getInputUnits() + 1);
        for(int j = 0; j < derivateWeightConnectInputAndHiddenMatrix.rowSize(); j++)
            derivateWeightConnectInputAndHiddenMatrix.viewRow(j).assign(derivateAj.get(j));

        //初始化连接隐藏层与输出层的权重的偏导数值.
        Matrix derivateWeightConnectHiddenAndOutPutMatrix = new DenseMatrix(labelMap.size(), netWorkModel.getM() + 1);
        for(int k = 0; k < derivateWeightConnectHiddenAndOutPutMatrix.rowSize(); k++)
            derivateWeightConnectHiddenAndOutPutMatrix.viewRow(k).assign(derivateAk.get(k));

        //由于隐藏层与输出层之间增加了一个bais的新增节点，而输入层与隐藏层的bais节点之间又没有联系.
        //因此，这里需要区分开来.
        Vector newFeatures = new DenseVector(netWorkModel.getInputUnits() + 1);
        newFeatures.set(0, 1.0);
        newFeatures.viewPart(1,netWorkModel.getInputUnits());
        derivateWeightConnectInputAndHiddenMatrix = derivateWeightsConnectLayers(derivateWeightConnectInputAndHiddenMatrix, newFeatures);
        Vector newZj = new DenseVector(netWorkModel.getM() + 1);
        newZj.set(0, 1.0);
        newZj.viewPart(1, netWorkModel.getM()).assign(Zj);
        derivateWeightConnectHiddenAndOutPutMatrix = derivateWeightsConnectLayers(derivateWeightConnectHiddenAndOutPutMatrix, newZj);
        updateModel(derivateWeightConnectInputAndHiddenMatrix, derivateWeightConnectHiddenAndOutPutMatrix);
//        return derivateAk.assign(new DoubleFunction() {
//            @Override
//            public double apply(double x) {
//                return Math.abs(x);
//            }
//        }).zSum();
        return labelMap.get(label) == yk.maxValueIndex();
    }

    public void updateModel(Matrix derivateWeightConnectInputAndHiddenMatrix, Matrix derivateWeightConnectHiddenAndOutPutMatrix){
        updateWeight(netWorkModel.getFirstLayerWeights(), derivateWeightConnectInputAndHiddenMatrix, lambda1);
        updateWeight(netWorkModel.getSecondLayerWeights(), derivateWeightConnectHiddenAndOutPutMatrix, lambda2);
    }

    /**更新权值.
     * @param layerWeights
     * @param derivateWeightConnectLayers
     * @param lambda
     */
    public void updateWeight(Matrix layerWeights, Matrix derivateWeightConnectLayers, double lambda){
        for(int row = 0; row < layerWeights.rowSize(); row++){
            Vector vector = layerWeights.viewRow(row).times(lambda);
            vector = (vector.plus(derivateWeightConnectLayers.viewRow(row))).times(learningRate);
            layerWeights.assignRow(row,layerWeights.viewRow(row).minus(vector));
        }
    }
    /**
     * 根据BP反向传播算法获得其En对隐藏层中aj的偏导数.
     * d(En/aj) = h'(aj) * sum(d(En/ak) * wkj) k = 1,2,...outputUnitNums;
     * @param Zj
     * @param derivateAK
     * @return
     */
    public Vector getDerivateAJ(Vector Zj, Vector derivateAK){

        //h'(aj) = h(aj) * (1 - h(aj)) = zj * ( 1 - zj)
        Vector derivateOfActivation = Zj.times(new DenseVector(netWorkModel.getM()).assign(1.0).minus(Zj));

        //sum(d(En/ak) * wkj)
        Vector fp = new DenseVector(netWorkModel.getM());
        for(int j = 0; j < fp.size(); j++){
            fp.set(j,(netWorkModel.getSecondLayerWeights().viewColumn(j).times(derivateAK)).zSum());
        }
        return derivateOfActivation.times(fp);
    }

    /**
     * 计算En对权重Wji的偏导数.用于SGD.
     * 注意这个结果是个矩阵.
     * d(E/Wji) = d(E/aj) * d(aj/Wji);
     * @param derivateA d(E/aj)
     * @param Z  d(aj/Wji)
     * @return
     */
    public Matrix derivateWeightsConnectLayers(Matrix derivateA, Vector Z){

        for(int row = 0; row < derivateA.rowSize(); row++)
            derivateA.assignRow(row,derivateA.viewRow(row).times(Z));
        return derivateA;
    }

    public HashMap<String,Integer> labelIndex(Path labelPath) throws IOException {
        HashMap<String,Integer> labelMap = new HashMap<>();
        BufferedReader reader = null;
        List<Path> listPaths = new ArrayList<>();
        paths(labelPath,listPaths);
        FileSystem fs = FileSystem.get(conf);
        for(Path path : listPaths){
            reader = new BufferedReader(new InputStreamReader(fs.open(path)));
            String str = null;
            while((str = reader.readLine()) != null){
                if(!labelMap.containsKey(str.trim()))
                    labelMap.put(str.trim(),labelMap.size());
            }
            reader.close();
        }

        return labelMap;
    }

    public void paths(Path path,List<Path> listPaths) throws IOException {
        for(FileStatus status : FileSystem.get(conf).listStatus(path)){
            if(status.isDirectory())
                paths(status.getPath(), listPaths);
            else
                listPaths.add(status.getPath());
        }
    }
}
