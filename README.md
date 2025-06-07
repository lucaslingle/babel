# Babel
Minimal LLM in Jax. 

![tower of babel](assets/babel_tower.png)

## ✨Features:
- Support for FSDP on TPU pods
- Support for LLaMA architecture
- Support for Distributed Muon
- Checkpointing with Orbax
- Logging with Weights and Biases
- Resumable training with NumPy Memmap

## 🚀 Getting started

#### 1. Install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh;
```

#### 2. Clone
```
git clone https://github.com/lucaslingle/babel.git;
cd babel;
```

#### 3. Install dependencies
```
uv sync;
```

#### 4. Run the launch script
```
uv run src/serious_mode/main.py \
    --config=src/babel/configs/MY_CONFIG.py \
    --workdir=MY_WORKDIR \
    --group=MY_EXPERIMENT_GROUP_NAME \
    --hf_token=MY_HF_TOKEN \
    --wb_token=MY_WB_TOKEN
```
- To override any setting in the config file, your can append ```--config.key=value``` to the command above.
- If you are overriding many configs, you can also write your own. For an example, see ```src/babel/configs/muon.py```. 
- For logging continuity in weights and biases, append ```--wb_runid=MY_RUNID```. 