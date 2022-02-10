import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashSet;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.stream.Collectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopkCommonWords {

    static HashSet<String> stopWords;
    static final int k = 20;

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, FileCountWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text file = new Text();
        private FileCountWritable fc = new FileCountWritable();
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String tmp = itr.nextToken();
                if (!stopWords.contains(tmp)) {
                    word.set(tmp);

                    String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
                    file.set(fileName);
                    fc.setFile(file);
                    fc.setCount(one);

                    context.write(word, fc);
                }
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text, FileCountWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<FileCountWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            HashMap<String, Integer> fileCountMap = new HashMap<>();

            for (FileCountWritable val : values) {
                String fileName = val.getFile().toString();
                int count = fileCountMap.getOrDefault(fileName, 0);
                fileCountMap.put(fileName, count + val.getCount().get());
            }
            if (fileCountMap.size() > 1) {
                result.set(Collections.max(fileCountMap.values()));
                context.write(key, result);
            }
        }
    }

    public static class TokenizerMapperWordCountInverted
            extends Mapper<Object, Text, WordCountKeyPair, IntWritable>{

        private Text word = new Text();
        private WordCountKeyPair pair = new WordCountKeyPair();
        private IntWritable count = new IntWritable();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            // Input is processed one line at a time,
            // meaning one word and corresponding word count separated by space
            String[] tokens = value.toString().split("\\s+");
            
            word.set(tokens[0]);
            pair.setWord(word);
            count.set(Integer.parseInt(tokens[1]));
            pair.setCount(count);
            
            context.write(pair, count);
        }
    }

    public static class IntSumReducerTopK
            extends Reducer<WordCountKeyPair, IntWritable, IntWritable, Text> {
        public void reduce(WordCountKeyPair key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            if (context.getCounter("org.apache.hadoop.mapred.Task$Counter",  "REDUCE_OUTPUT_RECORDS").getValue() < k) {
                        context.write(key.getCount(), key.getWord());
            }
        }
    }

    public static class FileCountWritable
            implements Writable {
        private Text file = new Text();
        private IntWritable count = new IntWritable();

        FileCountWritable() { }

        private void setFile(Text file) {
            this.file = file;
        }

        private void setCount(IntWritable count) {
            this.count = count;
        }

        private Text getFile() {
            return file;
        }

        private IntWritable getCount() {
            return count;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeUTF(file.toString());
            dataOutput.writeInt(count.get());
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            file.set(dataInput.readUTF());
            count.set(dataInput.readInt());
        }

        public FileCountWritable read(DataInput in) throws IOException {
            FileCountWritable w = new FileCountWritable();
            w.readFields(in);
            return w;
        }
    }

    public static class WordCountKeyPair
            implements Writable, WritableComparable<WordCountKeyPair> {
        private Text word = new Text();
        private IntWritable count = new IntWritable();

        WordCountKeyPair() { }

        @Override
        public int compareTo(WordCountKeyPair pair) {
            int compareValue = this.count.compareTo(pair.getCount());
            if (compareValue == 0) {
                compareValue = word.compareTo(pair.getWord());
            }
            return -compareValue;
        }

        private void setWord(Text word) {
            this.word = word;
        }

        private void setCount(IntWritable count) {
            this.count = count;
        }

        private Text getWord() {
            return word;
        }

        private IntWritable getCount() {
            return count;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeUTF(word.toString());
            dataOutput.writeInt(count.get());
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            word.set(dataInput.readUTF());
            count.set(dataInput.readInt());
        }

        public WordCountKeyPair read(DataInput in) throws IOException {
            WordCountKeyPair w = new WordCountKeyPair();
            w.readFields(in);
            return w;
        }
    }

    public static void main(String[] args) throws Exception {
        stopWords = Files.lines(Paths.get(args[2]))
                .collect(Collectors.toCollection(HashSet::new));

        Configuration confCommonGreater = new Configuration();
        Job jobCommonGreater = Job.getInstance(confCommonGreater, "word count 1");
        jobCommonGreater.setJarByClass(TopkCommonWords.class);
        jobCommonGreater.setMapperClass(TokenizerMapper.class);
        jobCommonGreater.setMapOutputKeyClass(Text.class);
        jobCommonGreater.setMapOutputValueClass(FileCountWritable.class);
        jobCommonGreater.setReducerClass(IntSumReducer.class);
        jobCommonGreater.setOutputKeyClass(Text.class);
        jobCommonGreater.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(jobCommonGreater, new Path(args[0]));
        FileInputFormat.addInputPath(jobCommonGreater, new Path(args[1]));
        FileOutputFormat.setOutputPath(jobCommonGreater, new Path("wccg"));
        jobCommonGreater.waitForCompletion(true);

        Configuration confTopK = new Configuration();
        Job jobTopK = Job.getInstance(confTopK, "top k");
        jobTopK.setJarByClass(TopkCommonWords.class);
        jobTopK.setMapperClass(TokenizerMapperWordCountInverted.class);
        jobTopK.setMapOutputKeyClass(WordCountKeyPair.class);
        jobTopK.setMapOutputValueClass(IntWritable.class);
        // No combiner as output of word count would not have repeated words
        jobTopK.setReducerClass(IntSumReducerTopK.class);
        jobTopK.setOutputKeyClass(IntWritable.class);
        jobTopK.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(jobTopK, new Path("wccg", "part-r-00000"));
        FileOutputFormat.setOutputPath(jobTopK, new Path(args[3]));
        System.exit(jobTopK.waitForCompletion(true) ? 0 : 1);
    }
}
