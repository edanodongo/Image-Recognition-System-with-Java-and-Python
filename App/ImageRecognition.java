import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.nio.file.Paths;

public class ImageRecognition {
    private SavedModelBundle model;

    public ImageRecognition(String modelPath) {
        model = SavedModelBundle.load(modelPath, "serve");
    }

    public float[] predict(float[] inputImage) {
        try (Tensor<Float> inputTensor = Tensor.create(new long[]{1, 32, 32, 3}, FloatBuffer.wrap(inputImage))) {
            Tensor<?> outputTensor = model.session().runner()
                .feed("serving_default_conv2d_input:0", inputTensor)
                .fetch("StatefulPartitionedCall:0")
                .run().get(0);

            float[][] output = new float[1][10];
            outputTensor.copyTo(output);

            return output[0];
        }
    }

    public static void main(String[] args) {
        ImageRecognition recognizer = new ImageRecognition("path/to/cifar10_model");
        float[] testImage = new float[32 * 32 * 3]; // Example test image
        float[] prediction = recognizer.predict(testImage);

        for (int i = 0; i < prediction.length; i++) {
            System.out.println("Class " + i + ": " + prediction[i]);
        }
    }
}