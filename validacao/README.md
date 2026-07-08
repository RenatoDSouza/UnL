# Validação

Esta pasta contém verificações rápidas para o pipeline experimental.

Objetivo:
- rodar smoke-tests do treino/unlearning;
- testar os encodings `angle`, `iqp` e `reupload`;
- detectar erros estruturais antes de executar o experimento completo no servidor.

Arquivos:
- `run_smoke_validation.py`: executa a validação mínima.
- `results.json`: saída gerada pelo smoke-test.
