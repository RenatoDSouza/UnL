# Validação

Esta pasta contém verificações rápidas para o pipeline experimental.

Objetivo:
- rodar smoke-tests do treino/unlearning com os dígitos 0 e 1 do FEMNIST;
- testar os encodings `angle`, `iqp` e `reupload`;
- detectar erros estruturais antes de executar o experimento completo no servidor.

Arquivos:
- `run_smoke_validation.py`: executa a validação mínima com `num_layers=2`,
  dois clientes e no máximo 12 exemplos por cliente.
- `results.json`: saída gerada pelo smoke-test.
