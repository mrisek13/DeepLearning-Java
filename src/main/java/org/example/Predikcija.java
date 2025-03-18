package org.example;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import java.nio.FloatBuffer;

public class Predikcija {

    public static void main(String[] args) {

        try (SavedModelBundle model = SavedModelBundle.load("saved_model/tourism_model")) {
            System.out.println("Model uspješno učitan!");

            float[] input = new float[]
             //  {1, 0, 0, 1, 0, 0, 1};// Muški, Zagreb, 20-30
            // {1, 0, 0, 1, 0, 0, 0}; // Muški, Zagreb, 30-40
            // {1, 0, 0, 0, 1, 0, 1}; // Muški, Varaždin, 20-30
            // {1, 0, 0, 0, 1, 0, 0}; // Muški, Varaždin, 30-40
            // {0, 1, 0, 1, 0, 0, 1}; // Ženski, Zagreb, 20-30
            // {0, 1, 0, 1, 0, 0, 0}; // Ženski, Zagreb, 30-40
            // {0, 1, 0, 0, 1, 0, 1}; // Ženski, Varaždin, 20-30
           // {0, 1, 0, 0, 0, 1, 0}; // Ženski, Osijek, 30-40
            {1, 0, 0, 0, 0, 1, 1}; // Muški, Osijek, 20-30

            // Kreiranje TensorFlow tenzora od inputa
            try (TFloat32 inputTensor = TFloat32.tensorOf(Shape.of(1, input.length))) {

                // Pokretanje predikcije
                Tensor outputTensor = model.session().runner()
                        .feed("serving_default_input_layer:0", inputTensor) // Ovo ime može varirati, provjeriti SavedModel signature
                        .fetch("StatefulPartitionedCall_1:0")  // Standardno ime izlaza, može se promijeniti!
                        .run()
                        .get(0);

                TFloat32 outputFloatTensor = (TFloat32) outputTensor;
                float[] output = new float[(int) outputTensor.shape().size(1)];
                final int[] index = {0}; // Indeks kao array jer lambda ne dozvoljava modifikaciju varijabli
                outputFloatTensor.scalars().forEach(scalar -> output[index[0]++] = scalar.getFloat());

                // Prikaz rezultata
                int predictedDestination = argmax(output, 0, 4);  // Prvih 4 indeksa su destinacije
                int predictedAccommodation = argmax(output, 4, 6);  // Zadnjih 4 su smještaj

                System.out.println("Predviđena destinacija: " + decodeDestination(predictedDestination));
                System.out.println("Predviđeni smještaj: " + decodeAccommodation(predictedAccommodation));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Funkcija za određivanje najveće vrijednosti (argmax)
    private static int argmax(float[] array, int start, int end) {
        int maxIndex = start;
        for (int i = start; i < end; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex - start; // Normalizacija indeksa
    }

    // Funkcije za dekodiranje predikcija u stvarne nazive
    private static String decodeDestination(int index) {
        String[] destinations = {"Pag", "Cres", "Pag", "Dubrovnik", "Pag", "Dubrovnik", "Pag", "Cres"};
        return destinations[index];
    }

    private static String decodeAccommodation(int index) {
        String[] accommodations = {"Apartman", "Hotel", "Apartman", "Hotel", "Kamp", "Hotel", "Kamp", "Hotel"};
        return accommodations[index];
    }
}