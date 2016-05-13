package neuralNetWork;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by sghipr on 5/11/16.
 * 将目标类别数据进行映射成一系列的数值.
 */
public class LabelIndexMap {

    public static class LabelIndexMapper extends Mapper<Text, VectorWritable, Text, NullWritable>{

        private Set<String> labelSet;
        public void setup(Context context){
            labelSet = new HashSet<>();
        }

        public void map(Text label, VectorWritable featureVector, Context context){
            labelSet.add(label.toString());
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            for(String label : labelSet)
                context.write(new Text(label), NullWritable.get());
        }
    }


    public static class LabelIndexReducer extends Reducer<Text, NullWritable, Text, NullWritable>{

        public void reduce(Text label, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
            context.write(label,NullWritable.get());
        }
    }
}
