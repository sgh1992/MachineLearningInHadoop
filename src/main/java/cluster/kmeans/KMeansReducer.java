package cluster.kmeans;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by sghipr on 5/19/16.
 */
public class KMeansReducer extends Reducer<IntWritable,Cluster,IntWritable,Cluster>{

    public void reduce(IntWritable clusterId, Iterable<Cluster> clusters, Context context) throws IOException, InterruptedException {

        Iterator<Cluster> iterator = clusters.iterator();
        Cluster sumCluster = new Cluster(iterator.next());
        while (iterator.hasNext()){
            sumCluster.addCluster(iterator.next());
        }
        sumCluster.computeParameters();
        context.write(clusterId, sumCluster);
    }
}