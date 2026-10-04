# Cheatsheet Rápido: Aprendizado Federado

**Assunto**: `aprendizado-federado`  
**Última Atualização**: 2026-10-04  

Arquivo de consulta rápida para comandos de ambiente, snippets de código essenciais e hiperparâmetros explorados durante o estudo de Redes Neurais (PyTorch) e Aprendizado Federado (Flower & NVFlare).

---

## 1. Comandos de Ambiente & Dependências

### Setup do Ambiente Virtual e Instalação Geral
```bash
# Criação e ativação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalação do PyTorch (com CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Pacotes para simulação federada, arquitetura e visualização
pip install "flwr[simulation]>=1.8.0" flwr-datasets nvflare
pip install numpy matplotlib torchviz torchinfo graphviz
```

---

## 2. Snippets de Código & Configurações Essenciais

### 2.1 Alocação de Dispositivo e Transferência Assíncrona
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aloca com pinned memory na CPU para transferência DMA rápida para a GPU
cpu_tensor = torch.randn(64, 3, 32, 32, pin_memory=(device.type == "cuda"))
gpu_tensor = cpu_tensor.to(device, non_blocking=True)
```

### 2.2 Zeramento Otimizado e Backward de Autograd
```python
# Otimização: set_to_none=True desaloca o tensor de gradientes e agiliza kernels CUDA
optimizer.zero_grad(set_to_none=True)

output = model(inputs)
loss = criterion(output, targets)
loss.backward()

# Gradient Clipping para estabilidade numérica
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 2.3 Retropropagação Não-Escalar (Produto Vetor-Jacobiano - VJP)
```python
# Quando a saída do modelo não é um escalar único, passe o vetor upstream v
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.stack([x[0]**2 + 2*x[1], 3*x[0] + x[1]**3])

# Vetor de sensibilidade v = [dL/dy1, dL/dy2]
v = torch.tensor([1.0, 1.0])
y.backward(gradient=v)
# x.grad conterá v^T * J
```

### 2.4 Modos de Desativação do Autograd
```python
# Inferência e Avaliação (Mínimo overhead de CPU e memória)
with torch.inference_mode():
    val_loss = criterion(model(val_inputs), val_targets)

# Atualização Manual de Pesos (Mantém version tracking, evita gravação no DAG)
with torch.no_grad():
    for param in model.parameters():
        if param.grad is not None:
            param -= learning_rate * param.grad
            param.grad = None

# Desacoplamento para Serialização (Retorna view desconectada do DAG)
detached_weight = param.detach()
```

### 2.5 Extração Segura de Pesos para Redes Federadas (Flower / NVFlare)
```python
def get_weights_numpy(model):
    """Extrai os pesos desconectando grafos e movendo para a CPU/NumPy."""
    with torch.inference_mode():
        return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_weights_numpy(model, weights_list):
    """Carrega parâmetros agregados do servidor de volta no modelo PyTorch."""
    state_dict = model.state_dict()
    for (k, _), w in zip(state_dict.items(), weights_list):
        state_dict[k] = torch.tensor(w)
    model.load_state_dict(state_dict)
```

### 2.6 Hook de Perturbação de Gradiente para Privacidade Diferencial (DP-SGD)
```python
def dp_hook(grad):
    clip_norm = 1.0
    noise_scale = 0.01
    norm = torch.norm(grad, p=2)
    clipped = grad * min(1.0, clip_norm / (norm + 1e-6))
    return clipped + torch.randn_like(clipped) * noise_scale

# Registra hook no parâmetro folha
hook_handle = param.register_hook(dp_hook)
# Para desativar: hook_handle.remove()
```

### 2.7 Bloco Convolucional Seguro para FL com GroupNorm (Substituto de BatchNorm)
```python
import torch.nn as nn

# Em redes federadas não-IID, substitua SEMPRE BatchNorm por GroupNorm
def build_conv_block_fl(in_channels, out_channels, num_groups=4):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        nn.GELU() # GELU elimina o fenômeno de neurônios mortos (Dying ReLU)
    )
```

### 2.8 Fórmulas e Função Utilitária de Dimensão Convolucional
```python
def calc_conv2d_dim(h_in, w_in, kernel_size=3, stride=1, padding=1, dilation=1):
    """Calcula dimensões espaciais exatas de saída após Conv2d ou Pooling."""
    h_out = ((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    w_out = ((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return h_out, w_out
```

### 2.9 Classificador com Logits Brutos e CrossEntropyLoss Estável
```python
# A camada linear final deve emitir LOGITS BRUTOS (sem Softmax prévio)
model = nn.Sequential(
    nn.Linear(512, 128),
    nn.GELU(),
    nn.Linear(128, num_classes) # Logits puros
)

# CrossEntropyLoss funde internamente LogSoftmax + NLLLoss via LogSumExp trick
criterion = nn.CrossEntropyLoss()
loss = criterion(model(inputs), targets)
```

### 2.10 Ponderação de Classes para Compensação de Dados Não-IID Locais
```python
# Compensação local quando o cliente possui classes altamente desbalanceadas
total_samples = sum(class_counts)
weights = [total_samples / (len(class_counts) * c) for c in class_counts]
class_weights = torch.tensor(weights, dtype=torch.float32)

criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
```

### 2.11 Diagnóstico de Parâmetros e Payload de Rede do Modelo em MB
```python
def get_model_payload_info(model):
    """Calcula contagem de parâmetros e tamanho em Megabytes para transmissão em FL."""
    total_params = sum(p.numel() for p in model.parameters())
    payload_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return {
        "params": total_params,
        "upload_mb": payload_mb,
        "roundtrip_mb": payload_mb * 2 # Download dos pesos globais + Upload dos pesos locais
    }
```

### 2.12 Fatiamento de Dataset para Clientes com `Subset` e `DataLoader`
```python
from torch.utils.data import Subset, DataLoader

# Fatiamento particionado sem duplicar tensores em memória
client_subset = Subset(full_dataset, client_indices)
client_loader = DataLoader(
    client_subset,
    batch_size=32,
    shuffle=True,        # Embaralha apenas as amostras locais
    pin_memory=True,     # Agiliza DMA para GPU
    num_workers=0,       # 0 evita overhead de spawn em clientes sequenciais
    drop_last=False      # Preserva 100% dos dados escassos do cliente
)
```

### 2.13 Função Canônica de Treino Local com Monitoramento de Vazão
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total += x.size(0)
        
    return {
        "loss": running_loss / total,
        "acc": (correct / total) * 100.0,
        "samples_per_sec": total / (time.time() - start_time + 1e-6)
    }
```

### 2.14 Função Canônica de Avaliação com `inference_mode`
```python
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            running_loss += criterion(logits, y).item() * x.size(0)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += x.size(0)
    return {"loss": running_loss / total, "acc": (correct / total) * 100.0}
```

### 2.15 Agregação Elementar de Pesos via `state_dict` (FedAvg)
```python
# Média ponderada ou aritmética de pesos no servidor central
def aggregate_fedavg(global_model, client_state_dicts, client_sample_counts):
    total_samples = sum(client_sample_counts)
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(
            client_state_dicts[i][key] * (client_sample_counts[i] / total_samples)
            for i in range(len(client_state_dicts))
        )
    global_model.load_state_dict(global_dict, strict=True)
```

---

## 3. Parâmetros, Hiperparâmetros & Dicas Práticas

### 3.1 Comparativo de Diretivas de Rastreamento de Gradiente

| Diretiva | Desativa Grafo? | Version Tracking | Recomendação de Uso em FL |
| :--- | :---: | :---: | :--- |
| `torch.no_grad()` | Sim | Ativo | Laço de otimização local manual de clientes. |
| `torch.inference_mode()` | Sim | Desativado | Avaliação de métricas locais e globais (validação). |
| `tensor.detach()` | Sim | Desacoplado | Conversão de pesos para listas/NumPy para envio ao agregador. |
| `requires_grad_(False)` | In-place | Ativo | Congelamento de backbone em fine-tuning federado. |

### 3.2 Prevenção de Falhas Críticas de Memória (CUDA OOM)

1. **Evite acúmulo de tensores de perda**:
   - ❌ **Errado**: `total_loss += loss` (mantém o DAG inteiro retido na memória).
   - ✅ **Correto**: `total_loss += loss.item()` (extrai apenas o escalar float primitivo).
2. **Adote `set_to_none=True`**:
   - `optimizer.zero_grad(set_to_none=True)` desaloca os tensores intermediários de gradiente em vez de preenchê-los com zeros, economizando memória de VRAM.
3. **Limpeza entre clientes federados**:
   - Ao rodar múltiplos clientes sequencialmente na mesma GPU, execute ao fim de cada cliente:
     ```python
     del model, optimizer
     import gc; gc.collect()
     if torch.cuda.is_available():
         torch.cuda.empty_cache()
     ```

### 3.3 Regras de Ouro Arquiteturais para Aprendizado Federado

1. **Banimento de `BatchNorm` em Dados Não-IID**:
   - `BatchNorm` acumula `running_mean` e `running_var` que divergem entre clientes heterogêneos. A média linear dessas estatísticas pelo FedAvg corrompe o modelo na avaliação global.
   - Utilize exclusivamente `GroupNorm` para visão computacional ou `LayerNorm` para Transformers/MLP.
2. **Substituição de `ReLU` por `GELU`**:
   - Evita o efeito *Dying ReLU*, onde taxas de aprendizado locais desativam neurônios permanentemente.
3. **Inicialização Global Idêntica na Rodada 0**:
   - É mandatório que o servidor central inicialize os pesos do modelo (`torch.manual_seed(42)`) e faça o broadcast dos mesmos pesos para todos os clientes na primeira rodada. Clientes iniciando de pesos aleatórios distintos não convergem no FedAvg.

### 3.4 Otimizadores Locais e Server Momentum (FedAvgM)

1. **Descarte de Estados de Momentum Local**:
   - Se o cliente federado retiver estados de momentum entre rodadas, ele acelerará na direção de seus dados locais privativos, agravando o *Client Drift*.
   - Use SGD com momentum zero localmente, ou reinicie `optimizer = SGD(...)` a cada rodada.
2. **Adoção de Server Momentum (FedAvgM)**:
   - Centralize o momentum no servidor agregador: acumule a inércia sobre o pseudo-gradiente médio global $\Delta \mathbf{w}_{avg}$. Isso garante aceleração global robusta sem polarizar os clientes.
