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

    /**
     * 获得其隐藏层中每个结点的值，其值通过输入层得到的.
     * 即aj = sum(wji * xi) i为输入层中结点的个数.
     *注意，为了便于理解，同时也为了方便后续接口的实现.
     * 这里features是原始特征向量的值，没有增加额外的偏置结点.
     * @param features 特征向量.
     * @return
     */
    public Vector hiddenUnitInput(Vector features){
        Vector vector = new DenseVector(inputUnits + 1);
        vector.set(0,1.0);
        vector.viewPart(1,inputUnits).assign(features);
        vector = firstLayerWeights.times(vector);
        return vector;
    }

    /**
     * 返回其隐藏层中每个结点，经过其激活函数之后的结果.
     * 激活函数默认为sigmoid函数.
     * @param features 特征向量.
     * @return
     */
    public Vector hiddenUnitActivate(Vector features){
        return hiddenUnitInput(features).assign(new DoubleFunction() {
            @Override
            public double apply(double x) {
                return 1.0/(1 + Math.exp(-x));
            }
        });
    }

    /**
     * 返回每个输出层中结点的值.ak
     * ak = sum(wkj * zj) j为隐藏层中结点的个数.
     * @param features
     * @return
     */
    public Vector outPutUnitInput(Vector features){

        Vector activateVector = hiddenUnitActivate(features);
        Vector vector = new DenseVector(M + 1);
        vector.set(0,1.0);
        vector.viewPart(1, M).assign(activateVector);
        return secondLayerWeights.times(vector);
    }
    /**
     * 注意输出层的激活函数是softMax的值.
     * 也就是其最终结果的预测值.
     * @param features
     * @return 返回的是各个类别的可能概率值.
     */
    public Vector predict(Vector features){
        Vector outputUnitInputs = outPutUnitInput(features);
        outputUnitInputs.assign(Functions.EXP);
        return outputUnitInputs.divide(outputUnitInputs.zSum());
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
        Random random = new Random(17);

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
        for(int i = 0; i < row; i++){
            Vector vector = new DenseVector(column);
            for(int j = 0; j < column; j++)
                vector.set(j,random.nextFloat());
            matrix.assignRow(i, vector);
        }
        return matrix;
    }
}
