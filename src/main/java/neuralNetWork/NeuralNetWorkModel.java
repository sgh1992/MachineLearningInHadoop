package neuralNetWork;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by sghipr on 5/11/16.
 * 神经网络的模型.
 * 这里默认为两层神经网络：即分别为输入层，隐藏层，输出层.
 * 多分类情况下的目标结果.
 * 考虑到偏置的影响，这里将明确：
 * firstLayerWeightMatrix    M,inputUnits + 1;
 * secondLayerWeightMatrix   outputUnits,M + 1;
 * M为真实的隐藏节点的个数，不包括bais节点.
 */
public class NeuralNetWorkModel implements Writable{

    /**
     * 输入层结点的个数.
     */
    private int inputUnits;

    /**
     * 输出层结点的个数.
     */
    private int outputUnits;

    /**
     * 输出层结点的个数.
     * 是一个可调的参数.
     */
    private int M;

    private Matrix firstLayerWeights;

    private Matrix secondLayerWeights;


    /**
     * 输出层结点的下标与其实际类别名称的映射值.
     * 0    ClassA
     * 1    ClassB
     * 2    ClassC
     */
    private MapWritable classLabelsMap;

    public void setM(int M){
        this.M = M;
    }

    public int getM(){
        return M;
    }

    public int getInputUnits(){
        return inputUnits;
    }

    public int getOutputUnits(){
        return outputUnits;
    }

    public Matrix getSecondLayerWeights(){
        return secondLayerWeights;
    }

    public Matrix getFirstLayerWeights(){
        return firstLayerWeights;
    }
    /**
     * just used as Serialized!
     */
    public NeuralNetWorkModel(){}

    @Override
    public void write(DataOutput out) throws IOException {

        out.writeInt(inputUnits);
        out.writeInt(M);
        out.writeInt(outputUnits);

        MatrixWritable.writeMatrix(out,firstLayerWeights);
        MatrixWritable.writeMatrix(out,secondLayerWeights);
        classLabelsMap.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {

        this.inputUnits = in.readInt();
        this.M = in.readInt();
        this.outputUnits = in.readInt();

        this.firstLayerWeights = MatrixWritable.readMatrix(in);
        this.secondLayerWeights = MatrixWritable.readMatrix(in);
        classLabelsMap.readFields(in);
    }


    public Vector forward(int layer,Vector insts){

        Vector addBias = new DenseVector(insts.size() + 1);
        addBias.set(0, 1);
        for(int i = 0; i < insts.size(); i++)
            addBias.set(i + 1,insts.get(i));
        Vector result = null;
        if(layer == 0)
            result = firstLayerWeights.times(addBias);
        else
            result = secondLayerWeights.times(addBias);
        return result.assign(Functions.SIGMOID);
    }

    public Vector predict(Vector features){

        for(int layer = 1; layer < 3; ++layer)
            features = forward(layer - 1,features);
        return features;
    }

    /**
     * 返回预测概率值最大的那个类别.
     * @param features
     * @return
     */
    public String classifyForInstance(Vector features){
        Vector result = predict(features);
        return classLabelsMap.get(new IntWritable(result.maxValueIndex())).toString();
    }
    /**
     * 初始化模型参数.
     * @param inputUnits
     * @param M
     * @param outputUnits
     */
    public void initializeModel(int inputUnits, int M, int outputUnits, HashMap<String,Integer> labelMap){

        this.inputUnits = inputUnits;
        this.outputUnits = outputUnits;
        setM(M);
        Random random = new Random();

        classLabelsMap = new MapWritable();
        for(Map.Entry<String,Integer> entry : labelMap.entrySet()){
            classLabelsMap.put(new IntWritable(entry.getValue()),new Text(entry.getKey()));
        }
        /**
         * 利用0-1之间的数值初始化神经网络的权值.
         */
        firstLayerWeights = initalMatrix(M,inputUnits + 1,random);
        secondLayerWeights = initalMatrix(outputUnits,M + 1,random);
    }

    private Matrix initalMatrix(int row,int column,Random random){

        Matrix matrix = new DenseMatrix(row, column);
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < column; ++j)
                matrix.set(i,j, random.nextDouble() - 0.5);
        }
        return matrix;
    }
}
