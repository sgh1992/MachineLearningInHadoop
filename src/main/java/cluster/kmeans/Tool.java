package cluster.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleFunction;
import java.io.IOException;
import java.util.*;

/**
 * Created by sghipr on 5/18/16.
 */
public class Tool {

    public static String initalModel = "initialModel";
    /**
     * 整个聚类结果的目录.
     */
    public static String clusterOutPut = "buildClusters";

    public static int featureNum(Path data, Configuration conf) throws IOException {

        SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), data, conf);
        IntWritable docId = (IntWritable)ReflectionUtils.newInstance(reader.getKeyClass(), conf);
        VectorWritable features = (VectorWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
        int numfeatures = 0;
        while (reader.next(docId, features)){
            numfeatures = features.get().size();
            break;
        }
        reader.close();
        return numfeatures;
    }


    /**
     * 随机初始化k个初始簇,每个簇向量的长度为length;
     * @param k 簇的个数.
     * @param length 向量的大小.
     * @return 返回初始化的簇.
     */
    public static List<Cluster> randomGenerator(int k, int length){

        Random random = new Random();
        List<Cluster> randomGroups = new ArrayList<>();
        for(int i = 0; i < k; ++i){
            Vector vector = new DenseVector(length);
            for(int j = 0; j < length; ++j)
                vector.set(j, random.nextGaussian());
            randomGroups.add(new Cluster(vector));
        }
        return randomGroups;
    }
    /**
     * @param data
     * @param K
     * @param conf
     * @return
     * @throws IOException
     */
    public static Path serializeModel(Path data, int K, Configuration conf) throws IOException {

        List<Cluster> clusters = randomGenerator(K, featureNum(data, conf));
        /**
         * 如果这里不存在clusterPath时,那么创建以clusterPath目录,以initalModel为子目录的新目录，会依次循环创建.
         *
         */
        Path clusterPath = new Path(data.getParent(), clusterOutPut);
        Path output = new Path(clusterPath, initalModel);
        SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.get(conf), conf, output, IntWritable.class, Cluster.class);
        for(int i = 0; i < clusters.size(); ++i)
            writer.append(new IntWritable(i), clusters.get(i));
        writer.close();
        return output;
    }

    public static void filePaths(Path root, List<Path> paths, FileSystem fileSystem) throws IOException {
        for(FileStatus status : fileSystem.listStatus(root)){
            if(status.isDirectory())
                filePaths(status.getPath(), paths, fileSystem);
            else
                paths.add(status.getPath());
        }
    }

    public static HashMap<Integer, Cluster> getClusters(Path clusterModels, Configuration conf) throws IOException {
        HashMap<Integer,Cluster> clusters = new HashMap<>();
        List<Path> paths = new ArrayList<>();
        FileSystem fileSystem = FileSystem.get(conf);
        Tool.filePaths(clusterModels, paths, fileSystem);
        SequenceFile.Reader reader = null;
        IntWritable clusterId;
        Cluster cluster = null;
        for(Path path : paths){
            reader = new SequenceFile.Reader(fileSystem, path, conf);
            clusterId = (IntWritable) ReflectionUtils.newInstance(reader.getKeyClass(),conf);
            cluster = (Cluster)ReflectionUtils.newInstance(reader.getValueClass(), conf);
            while (reader.next(clusterId,cluster)){
                clusters.put(clusterId.get(), new Cluster(cluster));
            }
            reader.close();
        }
        return clusters;
    }

    /**
     * 前后对比两个聚类模型，判断聚类结果是否收敛.
     * @param model1
     * @param model2
     * @param errorThreshold
     * @param configuration
     * @return
     * @throws IOException
     */
    public static boolean convergence(Path model1, Path model2, double errorThreshold, Configuration configuration) throws IOException {

        HashMap<Integer, Cluster> beforeModels = Tool.getClusters(model1, configuration);
        HashMap<Integer, Cluster> curModels = Tool.getClusters(model2, configuration);

        double sumError = 0.0;
        for(Map.Entry<Integer, Cluster> entry : beforeModels.entrySet()){
            if(curModels.containsKey(entry.getKey()))
                sumError += euclidDist(entry.getValue().getCenter(), curModels.get(entry.getKey()).getCenter());
            else {
                System.err.println("curClusters do not contains clusterID:" + entry.getKey());
                System.exit(1);
            }
        }
        if(sumError <= errorThreshold)
            return true;
        return false;
    }

    /**
     * 计算两个向量的欧几里得距离.
     * @param vector1
     * @param vector2
     * @return
     */
    public static double euclidDist(Vector vector1, Vector vector2){

        Vector dist = vector1.minus(vector2).assign(new DoubleFunction() {
            @Override
            public double apply(double x) {
                return x * x;
            }
        });
        return Math.sqrt(dist.zSum());
    }
}
