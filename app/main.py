from fastapi import FastAPI
import sne
import numpy as np
import time
import json

app = FastAPI()

@app.post("/sne")
async def stochastic(puntos: int, 
                     dim: int,
                     output_dim: int, 
                     perplexity: float,
                     learning_rate: float,
                     num_iterations: int):
    start = time.time()
    
    # Genera datos de ejemplo
    data = np.random.rand(puntos, dim).tolist()  # 20 puntos en 4 dimensiones

    # Inicializa el modelo SNE y ajusta
    #output_dim=2
    #perplexity=20.0
    #learning_rate=100.0
    #num_iterations=500

    model = sne.StochasticNeighborEmbedding(data, output_dim, perplexity, learning_rate)
    low_dim_data = model.fit(num_iterations)

    print("Representación en baja dimensión:", low_dim_data)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Representación en baja dimensión": low_dim_data
    }
    jj = json.dumps(j1)

    return jj