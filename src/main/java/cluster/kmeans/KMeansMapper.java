package cluster.kmeans;


import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by sghipr on 5/18/16.
 */
public class KMeansMapper extends Mapper<IntWritable, VectorWritable, IntWritable, Cluster> {

    private HashMap<Integer, Cluster> clusters;
    public void setup(Context context) throws IOException {
        clusters = Tool.getClusters(new Path(context.getCacheFiles()[0]), context.getConfiguration());
    }

    public void map(IntWritable docID, VectorWritable document, Context context){
        int bestCluster = cluster(document, clusters);
        if(bestCluster != -1)
            clusters.get(bestCluster).addPointToCluster(document.get());
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        for(Map.Entry<Integer,Cluster> entry : clusters.entrySet())
            context.write(new IntWritable(entry.getKey()), entry.getValue());
    }

    public int cluster(VectorWritable document, HashMap<Integer, Cluster> clusters){

        int bestCluster = -1;
        double minDist = Double.MAX_VALUE;
        for(Map.Entry<Integer, Cluster> entry : clusters.entrySet()){
            if(dist(entry.getValue().getCenter(),document.get()) < minDist){
                bestCluster = entry.getKey();
                minDist = dist(entry.getValue().getCenter(), document.get());
            }
        }
        return bestCluster;
    }

    /**
     *注意，这里使用的是余弦相似性的计算方式.
     * 两者越相似，则余弦计算出的结果越小.
     * 而这里统一采用的是一种基于距离的计算.
     * 距离越小者，则两者越相近.
     * 因此最终的结果是 1 - cosSimliarity
     * @param center
     * @param point
     * @return
     */
    public static double dist(Vector center, Vector point){

        if(center.size() != point.size()){
            System.err.println("center Vector and point Vector size do not equals in Similarity !");
            System.exit(1);
        }
        double square = center.dot(center) + point.dot(point);
        double cross = center.dot(point);
        if(cross > square)
            square = cross;
        if(square == 0)
            return 0;
        return 1 - cross/square;
    }

}
