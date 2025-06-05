package sutime;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.time.TimeAnnotations;
import comscsv.CommonsCSVReader;

import tempdata.*;
import java.util.*;
import java.util.List;
import java.util.Arrays;
import java.util.Properties;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import com.google.gson.Gson;
import java.io.FileWriter;
import java.io.IOException;
import com.google.gson.GsonBuilder;
import static comscsv.CommonsCSVReader.readCSV;

public class SUTimeBasicExample {

    public static String[] examples = {
            "2:00 17/6/2023",
            //"The meeting will be held at 4:00pm in the library",
            //"The conflict has lasted for over 15 years and shows no signs of abating."
    };
    public static void main(String[] args) {
        // set up pipeline properties
        Properties props = new Properties();
        // general properties
        props.setProperty("annotators", "tokenize,pos,lemma,ner");
        props.setProperty("ner.docdate.usePresent", "true");
        props.setProperty("sutime.includeRange", "true");
        props.setProperty("ner.rulesOnly", "true");
        props.setProperty("sutime.rules", "./defs.sutime.txt,./english.sutime.txt");
        props.setProperty("sutime.markTimeRanges", "true");
        // read csv file
        String path = "./test_cs.csv";
        List<String[]> data = readCSV(path);
        // build pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Map<Integer, TemporalData> temporalDataMap = new HashMap<>();

        for (int i = 1; i < data.size(); i++) {
            String[] example = data.get(i);
            CoreDocument document = new CoreDocument(example[0]);
            pipeline.annotate(document);

            for (CoreEntityMention cem : document.entityMentions()) {
                if (cem.coreMap().get(TimeAnnotations.TimexAnnotation.class) != null) {
                    String timex = cem.coreMap().get(TimeAnnotations.TimexAnnotation.class).toString();
                    Pattern pattern = Pattern.compile("type=\"([^\"]*)\"");
                    Matcher matcher = pattern.matcher(timex);
                    String type = null;
                    if (matcher.find()) {
                        type = matcher.group(1);
                    }
                    if (type != null) {
                        // Accumulate the data in the map
                        System.out.println("temporal expression: " + cem.text());
                        System.out.println("temporal type: " + type);
                        System.out.println("id: " + i);
                        temporalDataMap
                                .computeIfAbsent(i, k -> new TemporalData(k))
                                .addDetail(cem.text(), type);
                    }
                }
            }
        }

        // Sort the map by ID
        List<TemporalData> sortedData = new ArrayList<>(temporalDataMap.values());
        sortedData.sort(Comparator.comparingInt(TemporalData::getId));

        // Write the sorted data to a JSON file

        try (FileWriter writer = new FileWriter("output_test.json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(sortedData, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


