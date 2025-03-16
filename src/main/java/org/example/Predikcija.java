package org.example;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import java.nio.FloatBuffer;

public class Predikcija {

    public static void main(String[] args) {
        // Učitavanje modela iz direktorija "saved_model/tourism_model"
        try (SavedModelBundle model = SavedModelBundle.load("saved_model/tourism_model")) {
            System.out.println("Model uspješno učitan!");

            // Priprema ulaznih podataka: Muški, Zagreb, 20-30 (One-Hot Encoding)
            float[] input = new float[]{1, 0, 0, 1, 0, 0, 1, 0}; // One-hot encoded podaci

            // Kreiranje TensorFlow tenzora od inputa
            try (TFloat32 inputTensor = TFloat32.tensorOf(Shape.of(1, input.length))) {

                // Pokretanje predikcije
                Tensor outputTensor = model.session().runner()
                        .feed("serving_default_dense_input", inputTensor) // Ovo ime može varirati, provjeriti SavedModel signature
                        .fetch("StatefulPartitionedCall")  // Standardno ime izlaza, može se promijeniti!
                        .run()
                        .get(0);

                // Dobivanje rezultata
                float[] output = new float[(int) outputTensor.shape().size(1)];

                TFloat32 outputFloatTensor = (TFloat32) outputTensor;
                for (int i = 0; i < output.length; i++) {
                    output[i] = outputFloatTensor.getFloat(i);
                }

                // Prikaz rezultata
                int predictedDestination = argmax(output, 0, 4);  // Prvih 4 indeksa su destinacije
                int predictedAccommodation = argmax(output, 4, 8);  // Zadnjih 4 su smještaj

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
        String[] destinations = {"Pag", "Cres", "Dubrovnik", "Zagreb"};
        return destinations[index];
    }

    private static String decodeAccommodation(int index) {
        String[] accommodations = {"Apartman", "Hotel", "Kamp", "Hostel"};
        return accommodations[index];
    }
}