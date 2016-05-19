package cluster.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;

/**
 * Created by sghipr on 5/19/16.
 */
public class KMeansJob {

    public static Job kmeansJob(Path input, Path output,Path models, Configuration baseConf) throws IOException {

        Job job = Job.getInstance(baseConf);
        job.addCacheFile(models.toUri());

        job.setJarByClass(KMeansJob.class);

        job.setMapperClass(KMeansMapper.class);
        job.setCombinerClass(KMeansReducer.class);
        job.setReducerClass(KMeansReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Cluster.class);

        FileSystem.get(job.getConfiguration()).delete(output, true);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);
        return job;
    }

    public static Path buildClusters(Path data, Path model, int maxIters, double  errorThreshold, Configuration conf) throws IOException, ClassNotFoundException, InterruptedException {

        //first delete the outPut if it exists
        FileSystem fs = FileSystem.get(conf);
        Path clusters = new Path(data.getParent(), Tool.clusterOutPut);
//        fs.delete(clusters,true);
//        fs.mkdirs(clusters);

        Path output = null;
        int iter = 0;
        Job job = null;
        while (iter < maxIters){
            output = new Path(clusters, "cluster-" + iter);
            job = kmeansJob(data, output, model,conf);
            boolean success = job.waitForCompletion(true);
            if(!success){
                System.err.println("run buildClusters Job Failed!");
                System.exit(1);
            }

            if(Tool.convergence(model, FileOutputFormat.getOutputPath(job), errorThreshold,job.getConfiguration())){
                FileSystem.get(job.getConfiguration()).rename(output, new Path(clusters, "cluster-final"));
                return new Path(clusters, "cluster-final");
            }
            model = FileOutputFormat.getOutputPath(job);
            iter++;
        }
        FileSystem.get(job.getConfiguration()).rename(FileOutputFormat.getOutputPath(job), new Path(clusters, "cluster-final"));
        return FileOutputFormat.getOutputPath(job);
    }
}
