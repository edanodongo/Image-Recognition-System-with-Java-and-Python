package edan;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static float[] preprocessImage(String imagePath) {
        Mat image = Imgcodecs.imread(imagePath);
        Imgproc.resize(image, image, new Size(32, 32));
        image.convertTo(image, CvType.CV_32F);
        Core.divide(image, Scalar.all(255.0), image);

        float[] result = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, result);

        return result;
    }
}