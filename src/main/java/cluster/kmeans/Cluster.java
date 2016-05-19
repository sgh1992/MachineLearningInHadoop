package cluster.kmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by sghipr on 5/18/16.
 * 基于距离的聚类的簇群的数据结构.
 * 整个簇群的数据结构.
 */
public class Cluster implements Writable{

    private int numObservations;

    private Vector center;
    /**
     * 各个维度上的值之和.
     */
    private Vector S1;

    private Vector S2;

    /**
     * 每个簇的离散程度.以每个簇群的方差来代替.
     */
    private Vector std;

    public Cluster(Vector initalCenter){

        setCenter(initalCenter);
        std = new DenseVector(initalCenter.size());
        S1 = initalCenter.clone();
        S2 = initalCenter.like();
        numObservations = 1;

    }
    /**
     * @param other
     */
    public Cluster(Cluster other){
        S1 = new DenseVector(other.S1);
        S2 = new DenseVector(other.S2);
        center = new DenseVector(other.center);
        std = new DenseVector(other.std);
        this.numObservations = other.numObservations;
    }
    /**
     * just used as Serialized
     */
    public Cluster(){}

    public void setCenter(Vector center){
        this.center = center;
    }

    public void addPointToCluster(Vector point){
        S1.assign(point, Functions.PLUS);
        S2.assign(point.times(point), Functions.PLUS);
        numObservations++;
    }

    public void addCluster(Cluster cluster){

        S1.assign(cluster.S1,Functions.PLUS);
        S2.assign(cluster.S2,Functions.PLUS);
        this.numObservations += cluster.numObservations;
    }

    public void computeParameters(){

        if(numObservations != 0) {
            center = S1.divide(numObservations);
            //D(X) = E(X^2) - E(X) * E(X);
            std.assign(S2,Functions.PLUS).assign(numObservations);
            std.minus(S1.times(S1)).divide(numObservations * numObservations);
        }
        S1 = S1.like();
        S2 = S2.like();
        numObservations = 0;
    }

    public Vector getCenter(){
        return center;
    }

    @Override
    public void write(DataOutput out) throws IOException {

        out.writeInt(numObservations);
        VectorWritable.writeVector(out,S1);
        VectorWritable.writeVector(out,S2);
        VectorWritable.writeVector(out,center);
        VectorWritable.writeVector(out,std);

    }

    /**
     * 将结果全部写入到序列化文件中.
     * @param in
     * @throws IOException
     */
    @Override
    public void readFields(DataInput in) throws IOException {

        this.numObservations = in.readInt();
        this.S1 = VectorWritable.readVector(in);
        this.S2 = VectorWritable.readVector(in);
        this.center = VectorWritable.readVector(in);
        this.std = VectorWritable.readVector(in);
    }
}
