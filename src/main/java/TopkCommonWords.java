import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
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
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopkCommonWords {

    static HashSet<String> stopWords;
    static final int k = 20;

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String tmp = itr.nextToken();
                if (!stopWords.contains(tmp)) {
                    word.set(tmp);
                    context.write(word, one);
                }
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class TokenizerMapperWordCount
    extends Mapper<Object, Text, Text, IntWritable>{

        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            // Input is processed one line at a time,
            // meaning one word and corresponding word count separated by space
            String[] tokens = value.toString().split("\\s+");
            word.set(tokens[0]);
            context.write(word, new IntWritable(Integer.parseInt(tokens[1])));
        }
    }

    public static class IntSumReducerCommonGreater
    extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int cnt = 0;
            int greaterCommonCnt = 0;
            for (IntWritable val : values) {
                cnt++;
                int tmp = val.get();
                if (tmp > greaterCommonCnt) {
                    greaterCommonCnt = tmp;
                }
            }

            // If value occurs more than once, i.e. in more than one file, it is a common word
            if (cnt > 1) {
                result.set(greaterCommonCnt);
                context.write(key, result);
            }
        }
    }

    public static class TokenizerMapperWordCountInverted
            extends Mapper<Object, Text, CompositeKeyPair, IntWritable>{

        private CompositeKeyPair pair = new CompositeKeyPair();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            // Input is processed one line at a time,
            // meaning one word and corresponding word count separated by space
            String[] tokens = value.toString().split("\\s+");
            pair.setWord(new Text(tokens[0]));
            pair.setCount(new IntWritable(Integer.parseInt(tokens[1])));
            context.write(pair, new IntWritable());
        }
    }

    public static class IntSumReducerTopK
            extends Reducer<CompositeKeyPair, IntWritable, IntWritable, Text> {
        public void reduce(CompositeKeyPair key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            if (context.getCounter("org.apache.hadoop.mapred.Task$Counter",  "REDUCE_OUTPUT_RECORDS").getValue() < k) {
                        context.write(key.getCount(), key.getWord());
            }
        }
    }

    public static class CompositeKeyPair
            implements Writable, WritableComparable<CompositeKeyPair> {
        private Text word = new Text();
        private IntWritable count = new IntWritable();

        CompositeKeyPair() { }

        @Override
        public int compareTo(CompositeKeyPair pair) {
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
            dataOutput.writeInt(Integer.parseInt(count.toString()));
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            word.set(dataInput.readUTF());
            count.set(dataInput.readInt());
        }

        public CompositeKeyPair read(DataInput in) throws IOException {
            CompositeKeyPair w = new CompositeKeyPair();
            w.readFields(in);
            return w;
        }
    }

    public static void main(String[] args) throws Exception {
        stopWords = Files.lines(Paths.get(args[2]))
                .collect(Collectors.toCollection(HashSet::new));

        Configuration confWc1 = new Configuration();
        Job jobWc1 = Job.getInstance(confWc1, "word count 1");
        jobWc1.setJarByClass(TopkCommonWords.class);
        jobWc1.setMapperClass(TokenizerMapper.class);
        jobWc1.setCombinerClass(IntSumReducer.class);
        jobWc1.setReducerClass(IntSumReducer.class);
        jobWc1.setOutputKeyClass(Text.class);
        jobWc1.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(jobWc1, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobWc1, new Path("wc1"));
        jobWc1.waitForCompletion(true);

        Configuration confWc2 = new Configuration();
        Job jobWc2 = Job.getInstance(confWc2, "word count 2");
        jobWc2.setJarByClass(TopkCommonWords.class);
        jobWc2.setMapperClass(TokenizerMapper.class);
        jobWc2.setCombinerClass(IntSumReducer.class);
        jobWc2.setReducerClass(IntSumReducer.class);
        jobWc2.setOutputKeyClass(Text.class);
        jobWc2.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(jobWc2, new Path(args[1]));
        FileOutputFormat.setOutputPath(jobWc2, new Path("wc2"));
        jobWc2.waitForCompletion(true);

        Configuration confCommonGreater = new Configuration();
        Job jobCommonGreater = Job.getInstance(confCommonGreater, "common words");
        jobCommonGreater.setJarByClass(TopkCommonWords.class);
        jobCommonGreater.setMapperClass(TokenizerMapperWordCount.class);
        // No combiner as output of word count would not have repeated words
        jobCommonGreater.setReducerClass(IntSumReducerCommonGreater.class);
        jobCommonGreater.setOutputKeyClass(Text.class);
        jobCommonGreater.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(jobCommonGreater, new Path("wc1", "part-r-00000"));
        FileInputFormat.addInputPath(jobCommonGreater, new Path("wc2", "part-r-00000"));
        FileOutputFormat.setOutputPath(jobCommonGreater, new Path("wccg"));
        jobCommonGreater.waitForCompletion(true);

        Configuration confTopK = new Configuration();
        Job jobTopK = Job.getInstance(confTopK, "top k");
        jobTopK.setJarByClass(TopkCommonWords.class);
        jobTopK.setMapperClass(TokenizerMapperWordCountInverted.class);
        jobTopK.setMapOutputKeyClass(CompositeKeyPair.class);
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
