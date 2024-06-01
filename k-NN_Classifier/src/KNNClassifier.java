import java.util.*;

public class KNNClassifier {
    private int k;
    private List<DataPoint> trainData;

    public KNNClassifier(int k, List<DataPoint> trainData) {
        this.k = k;
        this.trainData = trainData;
    }

    public String classify(DataPoint newPoint){
        PriorityQueue<DataPointDistancePair> nearestNeighbours = new PriorityQueue<>(Collections.reverseOrder());

        for(DataPoint trainPoint : trainData){
            double distance = euclideanDistance(newPoint.getAttributes(), trainPoint.getAttributes());
            nearestNeighbours.add(new DataPointDistancePair(trainPoint, distance));
            if(nearestNeighbours.size() > k){
                nearestNeighbours.poll();
            }
        }
        return majorityVote(nearestNeighbours);
    }

    private double euclideanDistance(List<Double> attributes1, List<Double> attributes2){
        double sum = 0.0;
        for(int i = 0; i < attributes1.size(); i++){
            sum += Math.pow(attributes1.get(i) - attributes2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

    private String majorityVote(PriorityQueue<DataPointDistancePair> nearestNeighbours){
        Map<String, Integer> labelCounts = new HashMap<>();

        for(DataPointDistancePair pair : nearestNeighbours){
            String label = pair.dataPoint.getLabel();
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        String majorityLabel = null;
        int maxCount = -1;
        for(Map.Entry<String, Integer> entry : labelCounts.entrySet()){
            if(entry.getValue() > maxCount){
                maxCount = entry.getValue();
                majorityLabel = entry.getKey();
            }
        }
        return majorityLabel;
    }

    private class DataPointDistancePair implements Comparable<DataPointDistancePair> {
        private DataPoint dataPoint;
        private double distance;

        public DataPointDistancePair(DataPoint dataPoint, double distance) {
            this.dataPoint = dataPoint;
            this.distance = distance;
        }

        @Override
        public int compareTo(DataPointDistancePair o) {
            return Double.compare(this.distance, o.distance);
        }
    }
}
