# Paper

Este diretório contém o esqueleto do manuscrito em LaTeX para o projeto de Quantum Federated Learning com machine unlearning.

## Conteúdo

- `main.tex`: arquivo principal do paper
- `references.bib`: bibliografia inicial
- `sections/`: seções separadas para facilitar edição
- `figures/`: figuras do manuscrito
- `tables/`: tabelas do manuscrito
- `build/`: saída de compilação local

## Como compilar

Usando `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

Se preferir `pdflatex` e `bibtex` manualmente:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Escopo do manuscrito

O texto foi estruturado para um paper técnico com:

- introdução;
- metodologia;
- experimento de QFL;
- experimento de unlearning com QFI;
- métricas de avaliação;
- resultados esperados;
- discussão e limitações.

