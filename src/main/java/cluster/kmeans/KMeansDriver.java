package cluster.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

import java.io.IOException;
import java.util.List;

/**
 * Created by sghipr on 5/18/16.
 * KMeans聚类算法.
 * 分布式运行.
 * 其实这里除了理解并实现KMeans聚类算法之外，还需要理解,聚类算法的实际应用环境.
 * 因为一切的算法都是为了解决实际问题的.
 * 对任何算法都是如此.
 * 针对不同的环境，需要解决不同的问题.
 * 算法是一种手段，也是一种解决问题的思路.
 * 一切都是工具,只有能够解决实际问题才是最终目标!!!
 * 切记!!!
 *
 * 这里以经典的新闻数据来作为聚类数据.
 * 针对每个新闻文档，聚类出每篇新闻文档属于哪个类别.
 */
public class KMeansDriver extends Configured implements Tool{

    private static int K = 10;
    private static int maxIters = 30;
    private static double errorThreshold = 0.001;

    public KMeansDriver(int K, int maxIters, double errorThreshold){

        this.K = K;
        this.maxIters = maxIters;
        this.errorThreshold = errorThreshold;
    }

    @Override
    public int run(String[] args) throws Exception {
        return 0;
    }

    public Path runKmeansJob(Path data, Configuration conf) throws IOException, InterruptedException, ClassNotFoundException {

        Path model = cluster.kmeans.Tool.serializeModel(data,K,conf);
        Path clusters = KMeansJob.buildClusters(data,model,maxIters,errorThreshold,conf);
        return null;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new KMeansDriver(10,30,0.000001),args);
    }
}
