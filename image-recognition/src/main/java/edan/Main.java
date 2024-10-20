package edan;

public class Main {
    public static void main(String[] args) {
        ImagePreprocessor preprocessor = new ImagePreprocessor();
        ImageRecognition recognizer = new ImageRecognition("path/to/cifar10_model");

        float[] inputImage = preprocessor.preprocessImage("path/to/test_image.jpg");
        float[] prediction = recognizer.predict(inputImage);

        for (int i = 0; i < prediction.length; i++) {
            System.out.println("Class " + i + ": " + prediction[i]);
        }
    }
}