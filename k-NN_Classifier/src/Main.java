import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            List<DataPoint> trainData = CSVLoader.loadData("data/iris.data");
            List<DataPoint> testData = CSVLoader.loadData("data/iris.test.data");

            int k = 3;
            KNNClassifier knnClassifier = new KNNClassifier(k, trainData);

            int correct = 0;
            for(DataPoint testPoint : testData){
                String predictedLabel = knnClassifier.classify(testPoint);
                if(predictedLabel.equals(testPoint.getLabel())){
                    correct++;
                }
            }

            int total = testData.size();
            double accuracy = correct /(double) total;
            System.out.println("Correct test examples: " + correct);
            System.out.println("Total test examples: " + total);
            System.out.printf("Accuracy = %.2f = %.2f%%\n", accuracy, accuracy * 100);

            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            String input;
            while (true){
                System.out.println("Enter attributes for classification separated by commas (or type 'exit' to quit): ");
                input = br.readLine();
                if ("exit".equalsIgnoreCase(input)){
                    break;
                }
                List<Double> attributes = new ArrayList<>();
                for(String attribute : input.split(",")){
                    attributes.add(Double.parseDouble(attribute.trim()));
                }
                DataPoint newPoint = new DataPoint(attributes, null);
                String label = knnClassifier.classify(newPoint);
                System.out.println("Predicted label: " + label);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}