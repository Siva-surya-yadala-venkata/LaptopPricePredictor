import numpy as np

class RobustMLP:
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.001, dropout_rate=0.3, 
                 l2_lambda=0.0001, momentum=0.9):
        self.validate_parameters(input_size, hidden_layers, output_size)
        
        self.layers = [input_size] + hidden_layers + [output_size]
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        
        self.initialize_weights()
        
    def validate_parameters(self, input_size, hidden_layers, output_size):
        """Validate model parameters."""
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input and output sizes must be positive")
        if not all(size > 0 for size in hidden_layers):
            raise ValueError("Hidden layer sizes must be positive")
    
    def initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        self.weights = []
        self.biases = []
        self.weight_velocities = []
        self.bias_velocities = []
        
        for i in range(len(self.layers)-1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            
            W = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1])).astype(np.float64)
            b = np.zeros((1, self.layers[i+1]), dtype=np.float64)
            
            self.weights.append(W)
            self.biases.append(b)
            self.weight_velocities.append(np.zeros_like(W, dtype=np.float64))
            self.bias_velocities.append(np.zeros_like(b, dtype=np.float64))

    def leaky_relu(self, Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)

    def leaky_relu_derivative(self, Z, alpha=0.01):
        return np.where(Z > 0, 1, alpha)

    def dropout(self, A, training=True):
        if not training:
            return A
        mask = np.random.binomial(1, 1-self.dropout_rate, size=A.shape) / (1-self.dropout_rate)
        return A * mask

    def forward(self, X, training=True):
        if X.shape[1] != self.layers[0]:
            raise ValueError(f"Input shape {X.shape} does not match expected input size {self.layers[0]}")
            
        activations = [X.astype(np.float64)]
        pre_activations = []
        A = X.astype(np.float64)
        
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            pre_activations.append(Z)
            
            if i < len(self.weights) - 1:
                A = self.leaky_relu(Z)
                A = self.dropout(A, training)
            else:
                A = Z
            
            activations.append(A)
        
        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1).astype(np.float64)
        dW = []
        db = []
        
        dA = activations[-1] - y_true
        
        for i in reversed(range(len(self.weights))):
            dZ = dA * self.leaky_relu_derivative(pre_activations[i]) if i < len(self.weights) - 1 else dA
            
            dW_i = (np.dot(activations[i].T, dZ) / m) + (self.l2_lambda * self.weights[i])
            db_i = np.sum(dZ, axis=0, keepdims=True) / m
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
        
        return dW, db

    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weight_velocities[i] = (
                self.momentum * self.weight_velocities[i] - self.learning_rate * dW[i]
            )
            self.bias_velocities[i] = (
                self.momentum * self.bias_velocities[i] - self.learning_rate * db[i]
            )
            
            self.weights[i] += self.weight_velocities[i]
            self.biases[i] += self.bias_velocities[i]

    def train(self, X, y, X_val=None, y_val=None, epochs=1500, batch_size=64, early_stopping_patience=50):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            epoch_losses = []
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                activations, pre_activations = self.forward(X_batch)
                dW, db = self.backward(activations, pre_activations, y_batch)
                self.update_parameters(dW, db)
                
                batch_loss = np.mean((activations[-1] - y_batch.reshape(-1, 1))**2)
                epoch_losses.append(batch_loss)
            
            train_loss = np.mean(epoch_losses)
            training_history['train_loss'].append(train_loss)
            
            if X_val is not None:
                val_activations, _ = self.forward(X_val, training=False)
                val_loss = np.mean((val_activations[-1] - y_val.reshape(-1, 1))**2)
                training_history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")
        
        return training_history

    def predict(self, X):
        activations, _ = self.forward(X, training=False)
        return activations[-1]