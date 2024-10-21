package edan;

import org.tensorflow.SavedModelBundle;

public class ModelRead {
    public static void main(String[] args) {
        System.out.println("test load...");
        SavedModelBundle bundle = org.tensorflow.SavedModelBundle.load("C:/Users/Edan/OneDrive/Documents/GitHub/Image-Recognition-System-with-Java-and-Python/cifar10_model.keras","serve");
        System.out.println("loaded bundle...");
        System.out.println(bundle);
    }
}
