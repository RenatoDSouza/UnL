# QfederatedUlearning

Implementação base de **Quantum Federated Learning (QFL)** com foco em dois experimentos:

1. **Treinamento federado quântico puro**
   - Um servidor central coordena cinco clientes.
   - Os clientes treinam localmente um modelo quântico em `PennyLane`.
   - O servidor agrega as atualizações para produzir o modelo global.

2. **Machine unlearning com Quantum Fisher Information (QFI)**
   - O mesmo ambiente federado é treinado inicialmente.
   - Um cliente é desconectado da federação.
   - A influência histórica desse cliente é anulada por um fluxo baseado em QFI.

O repositório já está organizado para manter os dois experimentos em pastas separadas.

## Estrutura do projeto

```text
.
├── data/
├── experiments/
│   ├── qfl_training/
│   │   ├── configs/
│   │   ├── logs/
│   │   ├── outputs/
│   │   └── scripts/
│   └── qfl_unlearning_qfi/
│       ├── configs/
│       ├── logs/
│       ├── outputs/
│       └── scripts/
├── src/
│   └── qfl/
│       ├── common/
│       ├── data/
│       ├── federated/
│       ├── quantum/
│       └── utils/
├── tests/
├── pyproject.toml
└── README.md
```

## Stack técnica

- **Linguagem**: Python 3.10+
- **Biblioteca quântica**: `PennyLane`
- **Backend quântico preferencial**: `lightning.gpu`
- **Fallbacks automáticos**:
  - `lightning.qubit`
  - `default.qubit`
- **Bibliotecas auxiliares**:
  - `NumPy`
  - `PyTorch`
  - `pandas`
  - `scikit-learn`

## O que este projeto faz hoje

Nesta fase, o repositório contém:

- a arquitetura de servidor e cliente federado;
- o modelo quântico em `PennyLane`;
- a rotina de agregação do servidor;
- a rotina de unlearning baseada em QFI;
- scripts separados para cada experimento;
- testes mínimos para validar particionamento e agregação.

## Requisitos de ambiente

### 1. Python

Use Python 3.10 ou superior.

Verifique com:

```bash
python3 --version
```

### 2. GPU

O projeto tenta usar GPU via `lightning.gpu` quando disponível.

Importante:

- ter drivers NVIDIA compatíveis;
- ter CUDA corretamente instalado;
- ter uma versão de `PennyLane` e `pennylane-lightning` compatível com GPU;
- se a GPU não estiver disponível, o projeto faz fallback automático para CPU.

### 3. Ambiente virtual

É recomendável usar `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Instalação

Instale as dependências a partir do `requirements.txt`:

```bash
pip install -r requirements.txt
```

Se preferir instalar o pacote local em modo editável:

```bash
pip install -e .
```

## Dados

Os scripts atuais esperam um arquivo local:

```text
data/femnist_sample.npz
```

Esse arquivo deve conter duas chaves:

- `x`: imagens do FEMNIST com shape aproximado `(n_amostras, 28, 28)`
- `y`: rótulos associados

### Observação importante

O projeto ainda não inclui o downloader/parser completo do FEMNIST original do LEAF.
Portanto, nesta versão, você precisa disponibilizar manualmente um arquivo `npz` compatível para executar os scripts.

## Como preparar os dados para rodar agora

Se você já tiver um conjunto FEMNIST convertido para `npz`, coloque-o em:

```text
data/femnist_sample.npz
```

Se ainda não tiver esse arquivo, os scripts não vão executar até a etapa de ingestão ser implementada.

## Como rodar os experimentos

Os dois experimentos são independentes e ficam em diretórios distintos.

### 1. Experimento de treinamento QFL

Script:

```text
experiments/qfl_training/scripts/run_training.py
```

Execute com:

```bash
python3 experiments/qfl_training/scripts/run_training.py
```

O que esse script faz:

1. Carrega `data/femnist_sample.npz`
2. Normaliza as imagens
3. Reduz cada imagem para 4 features via quadrantes
4. Divide os dados entre 5 clientes
5. Executa 3 rodadas de treinamento federado
6. Salva um resumo em:

```text
experiments/qfl_training/outputs/training_summary.json
```

### 2. Experimento de machine unlearning com QFI

Script:

```text
experiments/qfl_unlearning_qfi/scripts/run_unlearning.py
```

Execute com:

```bash
python3 experiments/qfl_unlearning_qfi/scripts/run_unlearning.py
```

O que esse script faz:

1. Carrega `data/femnist_sample.npz`
2. Normaliza as imagens
3. Reduz cada imagem para 4 features via quadrantes
4. Divide os dados entre 5 clientes
5. Executa treinamento federado inicial
6. Remove o cliente `client_0`
7. Reexecuta a federação sem esse cliente
8. Calcula um score de QFI
9. Salva o resultado em:

```text
experiments/qfl_unlearning_qfi/outputs/unlearning_summary.json
```

## Configurações

Os valores padrão ficam nestes arquivos:

- `experiments/qfl_training/configs/default.yaml`
- `experiments/qfl_unlearning_qfi/configs/default.yaml`

Eles documentam parâmetros como:

- caminho do dataset;
- número de clientes;
- número de rodadas;
- cliente excluído no unlearning;
- preferência por GPU.

## Saída dos experimentos

### Treinamento

Saída esperada:

```json
{
  "rounds": [
    { "num_clients": 5.0 },
    { "num_clients": 5.0 },
    { "num_clients": 5.0 }
  ]
}
```

O formato pode evoluir conforme o treinamento quântico ficar mais sofisticado.

### Unlearning

Saída esperada:

```json
{
  "qfi_trace": 0.0,
  "remaining_clients": 4.0,
  "excluded_client": "client_0"
}
```

### Métricas de avaliação do unlearning

O experimento `qfl_unlearning_qfi` agora calcula três métricas principais:

#### 1. `forget_set_accuracy`

Mede a acurácia do modelo global sobre os dados do cliente removido da federação.

Interpretação desejada:

- quanto menor, melhor;
- idealmente deve se aproximar de uma taxa aleatória;
- indica que a influência histórica do cliente foi reduzida de forma efetiva.

#### 2. `retain_set_accuracy`

Mede a acurácia do modelo sobre os dados dos clientes que permanecem na federação.

Interpretação desejada:

- quanto maior, melhor;
- deve permanecer estável após o unlearning;
- evita `catastrophic forgetting` do conhecimento útil compartilhado.

#### 3. `mia_success_rate`

Mede a taxa de sucesso de um ataque de inferência de pertinência (`Membership Inference Attack`, MIA).

Interpretação desejada:

- quanto menor, melhor para privacidade;
- valores altos indicam maior risco de vazamento sobre o cliente removido;
- o projeto usa a biblioteca `Adversarial Robustness Toolbox (ART)` quando disponível.

Se `ART` não estiver instalado no ambiente, o projeto usa um proxy simples para manter o pipeline executável.

### Instalação opcional do ART

Para habilitar a integração com `ART`, instale o extra opcional:

```bash
pip install -e ".[art]"
```

Ou instale o pacote diretamente:

```bash
pip install adversarial-robustness-toolbox
```

Quando `ART` estiver presente, o experimento executa um ataque `MembershipInferenceBlackBox` em cima de um adaptador
`BlackBoxClassifier` construído a partir das probabilidades previstas pelo modelo quântico.
Sem `ART`, o código continua executando com uma aproximação interna para não interromper o fluxo.

#### 4. `qfi_trace`

Valor derivado da Quantum Fisher Information do modelo após o re-treinamento sem o cliente removido.

Interpretação:

- serve como indicador da sensibilidade do estado quântico;
- ajuda a comparar a magnitude da atualização antes e depois do unlearning.

### Exemplo de saída expandida

```json
{
  "qfi_trace": 0.0,
  "remaining_clients": 4.0,
  "excluded_client": "client_0",
  "forget_set_accuracy": 0.25,
  "retain_set_accuracy": 0.87,
  "random_baseline_accuracy": 0.25,
  "mia_success_rate": 0.50
}
```

## Testes

Há testes mínimos em:

```text
tests/test_federated.py
```

Execute com:

```bash
python3 -m pytest
```

Se o `pytest` ainda não estiver instalado:

```bash
pip install pytest
```

## Limitações atuais

Esta é uma base funcional, mas ainda não é a versão final de pesquisa.

Limitações atuais:

- o parser completo do FEMNIST do LEAF ainda não foi implementado;
- o treino quântico ainda usa uma aproximação simples;
- o fluxo de unlearning com QFI está estruturado, mas ainda é um ponto de partida metodológico;
- a integração com ART para MIA é efetiva quando a biblioteca está instalada, mas ainda mantém fallback quando a API do ambiente é incompatível;
- o formato `npz` é uma camada intermediária para permitir execução rápida enquanto o ingest pipeline definitivo não entra.

## Próximos passos recomendados

1. Implementar o download e parsing do FEMNIST original do LEAF.
2. Substituir a heurística de treino por otimização variacional real em `PennyLane`.
3. Formalizar o pipeline de unlearning com QFI com métricas mais robustas.
4. Adicionar CLI para execução parametrizada via YAML.
5. Expandir a suíte de testes.

## Resumo rápido

Se você já tem o dataset preparado:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python3 experiments/qfl_training/scripts/run_training.py
python3 experiments/qfl_unlearning_qfi/scripts/run_unlearning.py
```

Se você ainda não tem o arquivo `data/femnist_sample.npz`, primeiro precisa gerar esse dataset intermediário.
