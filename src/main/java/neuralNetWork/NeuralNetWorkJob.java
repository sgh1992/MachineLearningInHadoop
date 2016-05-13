package neuralNetWork;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

/**
 * Created by sghipr on 5/11/16.
 */
public class NeuralNetWorkJob {

    private static String labelIndexOutPut = "labelIndex";

    public static Job labelIndexJob(Configuration baseConf, Path trainDatas) throws IOException {

        Job job = Job.getInstance(baseConf);
        job.setJarByClass(NeuralNetWorkJob.class);

        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputValueClass(TextOutputFormat.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(NullWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);

        job.setMapperClass(LabelIndexMap.LabelIndexMapper.class);
        job.setReducerClass(LabelIndexMap.LabelIndexReducer.class);
        job.setCombinerClass(LabelIndexMap.LabelIndexReducer.class);

        Path output = new Path(trainDatas.getParent(), labelIndexOutPut);
        FileSystem.get(job.getConfiguration()).delete(output,true);
        FileInputFormat.addInputPath(job,trainDatas);
        FileOutputFormat.setOutputPath(job, output);
        return job;
    }

    public static Path runLabelIndexJob(Configuration baseConf, Path trainDatas) throws IOException, ClassNotFoundException, InterruptedException {
        Job job = labelIndexJob(baseConf, trainDatas);
        boolean success = job.waitForCompletion(true);
        if(!success){
            System.err.println("RunLabelIndexJob failed");
            System.exit(1);
        }
        return FileOutputFormat.getOutputPath(job);
    }
}
