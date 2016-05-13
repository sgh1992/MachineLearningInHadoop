package classify;
import org.apache.hadoop.fs.Path;

/**
 * Created by sghipr on 5/11/16.
 *所有分类模型的模板类.
 *一般包含两个功能.
 * 1.建立分类器. buildClassify.
 * 2.用构建的分类器，进行分类.classifyForInstance.
 */
public abstract class Classify {

    /**
     * 根据所给的文件作为其训练集，来训练其分类器.
     * 更一般的做法是用一个表征所有数据的Matrix来作为其训练集.
     * @param files
     */
    public abstract void buildClassify(Path files);


}
