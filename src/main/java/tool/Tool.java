package tool;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by sghipr on 15/05/16.
 */
public class Tool {

    public static HashMap<String,Integer> labelIndex(Path labelPath, Configuration conf) throws IOException {
        HashMap<String,Integer> labelMap = new HashMap<>();
        BufferedReader reader = null;
        List<Path> listPaths = new ArrayList<>();
        paths(labelPath,listPaths,conf);
        FileSystem fs = FileSystem.get(conf);
        for(Path path : listPaths){
            reader = new BufferedReader(new InputStreamReader(fs.open(path)));
            String str = null;
            while((str = reader.readLine()) != null){
                if(!labelMap.containsKey(str.trim()))
                    labelMap.put(str.trim(),labelMap.size());
            }
            reader.close();
        }

        return labelMap;
    }

    public static void paths(Path path,List<Path> listPaths, Configuration conf) throws IOException {
        for(FileStatus status : FileSystem.get(conf).listStatus(path)){
            if(status.isDirectory())
                paths(status.getPath(), listPaths,conf);
            else
                listPaths.add(status.getPath());
        }
    }
}
