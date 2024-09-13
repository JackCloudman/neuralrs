use rand::Rng;
use std::f64::consts::E;

// Función de activación Sigmoid
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

// Derivada de la función Sigmoid, tomando como entrada la salida de sigmoid(x)
fn sigmoid_derivative(s: f64) -> f64 {
    s * (1.0 - s)
}

// Estructura de la Red Neuronal
struct NeuralNetwork {
    // Pesos de la capa de entrada a la capa oculta
    weights_input_hidden: Vec<Vec<f64>>,
    // Pesos de la capa oculta a la capa de salida
    weights_hidden_output: Vec<f64>,
    // Sesgos de la capa oculta
    bias_hidden: Vec<f64>,
    // Sesgo de la capa de salida
    bias_output: f64,
    // Tasa de aprendizaje
    learning_rate: f64,
}

impl NeuralNetwork {
    // Inicialización de la red neuronal
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();

        // Inicializar pesos de entrada a oculta
        let weights_input_hidden = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        // Inicializar pesos de oculta a salida
        let weights_hidden_output = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Inicializar sesgos
        let bias_hidden = vec![rng.gen_range(-1.0..1.0); hidden_size];
        let bias_output = rng.gen_range(-1.0..1.0);

        NeuralNetwork {
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            learning_rate,
        }
    }

    // Forward Pass
    fn forward(&self, inputs: &Vec<f64>) -> (Vec<f64>, f64) {
        // Capa Oculta
        let hidden_inputs: Vec<f64> = self.weights_input_hidden.iter()
            .enumerate()
            .map(|(i, weights)| {
                weights.iter().zip(inputs.iter())
                    .map(|(w, input)| w * input)
                    .sum::<f64>() + self.bias_hidden[i] // Usar el sesgo correspondiente
            })
            .collect();

        let hidden_outputs: Vec<f64> = hidden_inputs.iter().map(|x| sigmoid(*x)).collect();

        // Capa de Salida
        let final_input: f64 = self.weights_hidden_output.iter()
            .zip(hidden_outputs.iter())
            .map(|(w, h)| w * h)
            .sum::<f64>() + self.bias_output;

        let final_output = sigmoid(final_input);

        (hidden_outputs, final_output)
    }

    // Backward Pass y actualización de pesos
    fn train(&mut self, inputs: &Vec<f64>, target: f64) {
        // Forward pass
        let (hidden_outputs, final_output) = self.forward(inputs);

        // Calcular error en la salida
        let output_error = final_output - target;
        let output_delta = output_error * sigmoid_derivative(final_output);

        // Calcular error en la capa oculta
        let hidden_errors: Vec<f64> = self.weights_hidden_output.iter()
            .map(|w| w * output_delta)
            .collect();

        let hidden_deltas: Vec<f64> = hidden_errors.iter()
            .zip(hidden_outputs.iter())
            .map(|(e, h)| e * sigmoid_derivative(*h))
            .collect();

        // Actualizar pesos de oculta a salida
        for i in 0..self.weights_hidden_output.len() {
            self.weights_hidden_output[i] -= self.learning_rate * output_delta * hidden_outputs[i];
        }

        // Actualizar sesgo de salida
        self.bias_output -= self.learning_rate * output_delta;

        // Actualizar pesos de entrada a oculta
        for i in 0..self.weights_input_hidden.len() {
            for j in 0..self.weights_input_hidden[i].len() {
                self.weights_input_hidden[i][j] -= self.learning_rate * hidden_deltas[i] * inputs[j];
            }
            // Actualizar sesgo de la capa oculta
            self.bias_hidden[i] -= self.learning_rate * hidden_deltas[i];
        }
    }

    // Predicción
    fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let (_, final_output) = self.forward(inputs);
        final_output
    }
}

fn main() {
    // Crear una red neuronal con 2 entradas, 2 neuronas ocultas y 1 salida
    let mut nn = NeuralNetwork::new(2, 2, 1, 0.5);

    // Datos de entrenamiento para el problema XOR
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    // Entrenamiento
    for epoch in 0..10000 {
        for (inputs, target) in &training_data {
            nn.train(&inputs, *target);
        }

        // Opcional: imprimir el error cada 1000 épocas
        if epoch % 1000 == 0 {
            let mut total_error = 0.0;
            for (inputs, target) in &training_data {
                let output = nn.predict(&inputs);
                let error = (output - target).powi(2);
                total_error += error;
            }
            println!("Época {}: Error = {}", epoch, total_error);
        }
    }

    // Pruebas después del entrenamiento
    println!("\nResultados después del entrenamiento:");
    for (inputs, target) in &training_data {
        let output = nn.predict(&inputs);
        println!("Entrada: {:?}, Predicción: {:.4}, Objetivo: {}", inputs, output, target);
    }
}
