# QfederatedUlearning

Projeto base para Quantum Federated Learning (QFL) com dois experimentos separados:

1. Treinamento puro de modelo global em uma federação com um servidor central e cinco clientes.
2. Machine unlearning com desconexão de um cliente e remoção da sua influência histórica via Quantum Fisher Information (QFI).

## Organização

```text
.
├── experiments
│   ├── qfl_training
│   │   ├── configs
│   │   ├── logs
│   │   ├── outputs
│   │   └── scripts
│   └── qfl_unlearning_qfi
│       ├── configs
│       ├── logs
│       ├── outputs
│       └── scripts
├── src
│   └── qfl
│       ├── common
│       ├── data
│       ├── federated
│       ├── quantum
│       └── utils
└── README.md
```

## Objetivo arquitetural

- `src/qfl/common`: contratos, tipos, validações e constantes compartilhadas.
- `src/qfl/data`: ingestão, particionamento e pré-processamento do FEMNIST.
- `src/qfl/federated`: abstrações do servidor central, clientes e estratégia de agregação.
- `src/qfl/quantum`: circuito quântico, encoder, métrica e rotinas relacionadas a QFI.
- `src/qfl/utils`: logging, seed, serialização e helpers de engenharia.
- `experiments/qfl_training`: fluxo do experimento de QFL puro.
- `experiments/qfl_unlearning_qfi`: fluxo do experimento de unlearning baseado em QFI.

## Próximos passos

1. Definir a stack exata do projeto, por exemplo `PennyLane` ou `Qiskit`.
2. Implementar o modelo quântico, a estratégia de agregação e o particionamento FEMNIST.
3. Adicionar testes unitários e integração mínima para os dois experimentos.
